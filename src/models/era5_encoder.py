import torch
import torch.nn as nn


# ============================================================
# CONVLSTM CELL
# ============================================================

class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3,
    ):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
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

        c_next = (
            f * c
            + i * g
        )

        h_next = (
            o * torch.tanh(c_next)
        )

        return h_next, c_next


# ============================================================
# ERA5 ATMOSPHERIC ENCODER
# ============================================================

class ERA5AtmosphericEncoder(nn.Module):

    def __init__(
        self,
        input_channels=7,
        hidden_channels=64,
    ):

        super().__init__()

        self.hidden_channels = hidden_channels

        # First spatio-temporal layer
        self.encoder1 = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )

        # Second deeper layer
        self.encoder2 = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )

        # Feature projection
        self.projection = nn.Sequential(

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
            ),
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        """
        Input:
            x = (B, T, 7, H, W)

        Output:
            latent = (B, 64, H, W)
        """

        batch_size = x.size(0)

        height = x.size(3)
        width = x.size(4)

        device = x.device

        # ----------------------------------------------------
        # Initial states
        # ----------------------------------------------------

        h1 = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
        )

        c1 = torch.zeros_like(h1)

        h2 = torch.zeros_like(h1)
        c2 = torch.zeros_like(h1)

        # ----------------------------------------------------
        # Temporal encoding
        # ----------------------------------------------------

        for t in range(x.size(1)):

            frame = x[:, t]

            h1, c1 = self.encoder1(
                frame,
                h1,
                c1,
            )

            h2, c2 = self.encoder2(
                h1,
                h2,
                c2,
            )

        # ----------------------------------------------------
        # Latent representation
        # ----------------------------------------------------

        latent = self.projection(h2)

        return latent


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MODEL B — ERA5 ENCODER SANITY TEST")
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
    # Create model
    # --------------------------------------------------------

    model = ERA5AtmosphericEncoder(
        input_channels=7,
        hidden_channels=64,
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
    # Fake ERA5 batch
    #
    # B = 2
    # T = 12
    # C = 7
    # H = 20
    # W = 30
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

    with torch.no_grad():

        latent = model(x)

    print(
        "Input shape :",
        tuple(x.shape)
    )

    print(
        "Latent shape:",
        tuple(latent.shape)
    )

    print(
        "Latent min  :",
        latent.min().item()
    )

    print(
        "Latent max  :",
        latent.max().item()
    )

    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert latent.shape == (
        2,
        64,
        20,
        30,
    )

    assert torch.isfinite(
        latent
    ).all()

    print()
    print("=" * 60)
    print("STATUS: ERA5 ENCODER SANITY TEST PASSED")
    print("=" * 60)