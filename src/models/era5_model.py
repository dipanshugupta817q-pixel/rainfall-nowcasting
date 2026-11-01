import torch
import torch.nn as nn

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from models.era5_encoder import ERA5AtmosphericEncoder


# ============================================================
# ERA5 TEMPORAL DECODER CELL
# ============================================================

class ERA5DecoderCell(nn.Module):

    def __init__(
        self,
        input_channels=7,
        hidden_channels=64,
    ):

        super().__init__()

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, h, c):

        combined = torch.cat(
            [x, h],
            dim=1
        )

        gates = self.conv(combined)

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# ============================================================
# ERA5 FORECAST MODEL
# ============================================================

class ERA5ForecastModel(nn.Module):

    def __init__(
        self,
        input_channels=7,
        hidden_channels=64,
        output_channels=7,
        forecast_steps=12,
    ):

        super().__init__()

        self.hidden_channels = hidden_channels
        self.forecast_steps = forecast_steps

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.encoder = ERA5AtmosphericEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder = ERA5DecoderCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
        )

        # ----------------------------------------------------
        # Initial decoder state projection
        # ----------------------------------------------------

        self.hidden_projection = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.cell_projection = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
        )

        # ----------------------------------------------------
        # Output head
        # ----------------------------------------------------

        self.output_head = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                output_channels,
                kernel_size=1,
            ),
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        """
        Input:
            x = (B, 12, 7, H, W)

        Output:
            y = (B, 12, 7, H, W)
        """

        batch_size = x.size(0)

        height = x.size(3)
        width = x.size(4)

        # ----------------------------------------------------
        # Encode observed atmospheric history
        # ----------------------------------------------------

        latent = self.encoder(x)

        # ----------------------------------------------------
        # Initialize decoder state
        # ----------------------------------------------------

        h = self.hidden_projection(
            latent
        )

        c = self.cell_projection(
            latent
        )

        # ----------------------------------------------------
        # Start decoder from latest observed ERA5 frame
        # ----------------------------------------------------

        previous_frame = x[:, -1]

        predictions = []

        # ----------------------------------------------------
        # Autoregressive forecasting
        # ----------------------------------------------------

        for _ in range(self.forecast_steps):

            h, c = self.decoder(
                previous_frame,
                h,
                c,
            )

            prediction = self.output_head(
                h
            )

            predictions.append(
                prediction
            )

            # Feed prediction into next timestep
            previous_frame = prediction

        # ----------------------------------------------------
        # (B,T,C,H,W)
        # ----------------------------------------------------

        output = torch.stack(
            predictions,
            dim=1
        )

        return output


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL B — ERA5 FORECAST MODEL SANITY TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ERA5ForecastModel(
        input_channels=7,
        hidden_channels=64,
        output_channels=7,
        forecast_steps=12,
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Fake normalized ERA5 input
    # --------------------------------------------------------

    x = torch.randn(
        2,
        12,
        7,
        20,
        30,
        device=device,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    output = model(x)

    print(
        "Input shape :",
        tuple(x.shape)
    )

    print(
        "Output shape:",
        tuple(output.shape)
    )

    print(
        "Output min  :",
        output.min().item()
    )

    print(
        "Output max  :",
        output.max().item()
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert output.shape == (
        2,
        12,
        7,
        20,
        30,
    )

    assert torch.isfinite(
        output
    ).all()

    print()
    print("=" * 60)
    print("STATUS: ERA5 FORECAST MODEL SANITY TEST PASSED")
    print("=" * 60)