import torch
import torch.nn as nn


# ============================================================
# ConvLSTM Cell
# ============================================================

class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size=3
    ):

        super().__init__()

        padding = kernel_size // 2

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(
            in_channels=(
                input_channels
                + hidden_channels
            ),
            out_channels=(
                4 * hidden_channels
            ),
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(
        self,
        x,
        state
    ):

        h_prev, c_prev = state

        combined = torch.cat(
            [x, h_prev],
            dim=1
        )

        gates = self.conv(
            combined
        )

        i, f, o, g = torch.chunk(
            gates,
            4,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = (
            f * c_prev
            + i * g
        )

        h = (
            o
            * torch.tanh(c)
        )

        return h, c


# ============================================================
# Rainfall ConvLSTM
# ============================================================

class RainfallConvLSTM(nn.Module):

    def __init__(
        self,
        input_channels=1,
        hidden_channels=64,
        output_channels=1,
        input_frames=12,
        output_frames=12
    ):

        super().__init__()

        self.input_frames = input_frames

        self.output_frames = output_frames

        self.hidden_channels = hidden_channels

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        self.encoder1 = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=3
        )

        self.encoder2 = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        self.decoder1 = ConvLSTMCell(
            input_channels=output_channels,
            hidden_channels=hidden_channels,
            kernel_size=3
        )

        self.decoder2 = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3
        )

        # ----------------------------------------------------
        # OUTPUT HEAD
        # ----------------------------------------------------

        self.output_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=output_channels,
            kernel_size=1
        )

        # Rainfall in log1p space must be >= 0.
        self.output_activation = nn.Softplus()

    # ========================================================
    # STATE INITIALIZATION
    # ========================================================

    def _init_state(
        self,
        batch_size,
        height,
        width,
        device,
        dtype
    ):

        h = torch.zeros(
            batch_size,
            self.hidden_channels,
            height,
            width,
            device=device,
            dtype=dtype
        )

        c = torch.zeros_like(h)

        return h, c

    # ========================================================
    # ENCODER
    # ========================================================

    def encode(
        self,
        x,
        return_state=False
    ):
        """
        Input:
            x = (B, T, H, W, C)

        Returns:
            latent features = (B, hidden, H, W)

        If return_state=True:
            returns encoder hidden/cell states too.
        """

        batch_size = x.shape[0]

        height = x.shape[2]

        width = x.shape[3]

        device = x.device

        dtype = x.dtype

        # Convert:
        # (B,T,H,W,C)
        # ->
        # (B,T,C,H,W)

        x = x.permute(
            0,
            1,
            4,
            2,
            3
        )

        h1, c1 = self._init_state(
            batch_size,
            height,
            width,
            device,
            dtype
        )

        h2, c2 = self._init_state(
            batch_size,
            height,
            width,
            device,
            dtype
        )

        # ----------------------------------------------------
        # Process observed sequence
        # ----------------------------------------------------

        for t in range(
            self.input_frames
        ):

            frame = x[:, t]

            h1, c1 = self.encoder1(
                frame,
                (h1, c1)
            )

            h2, c2 = self.encoder2(
                h1,
                (h2, c2)
            )

        if return_state:

            return (
                h2,
                c2
            )

        return h2

    # ========================================================
    # DECODER
    # ========================================================

    def decode(
        self,
        latent,
        initial_frame
    ):
        """
        Autoregressive temporal decoder.

        latent:
            (B, hidden, H, W)

        initial_frame:
            Last observed rainfall frame
            (B, H, W, C)

        Returns:
            (B, output_frames, H, W, C)
        """

        batch_size = latent.shape[0]

        height = latent.shape[2]

        width = latent.shape[3]

        # ----------------------------------------------------
        # Initialize decoder states from encoder latent
        # ----------------------------------------------------

        h1 = latent

        c1 = torch.zeros_like(
            latent
        )

        h2 = latent

        c2 = torch.zeros_like(
            latent
        )

        # ----------------------------------------------------
        # Convert initial frame
        # ----------------------------------------------------

        previous_frame = initial_frame.permute(
            0,
            3,
            1,
            2
        )

        predictions = []

        # ----------------------------------------------------
        # Autoregressive forecasting
        # ----------------------------------------------------

        for _ in range(
            self.output_frames
        ):

            h1, c1 = self.decoder1(
                previous_frame,
                (h1, c1)
            )

            h2, c2 = self.decoder2(
                h1,
                (h2, c2)
            )

            prediction = self.output_conv(
                h2
            )

            prediction = self.output_activation(
                prediction
            )

            predictions.append(
                prediction
            )

            # Next decoder input =
            # current prediction

            previous_frame = prediction

        # ----------------------------------------------------
        # Stack
        # ----------------------------------------------------

        predictions = torch.stack(
            predictions,
            dim=1
        )

        # Current:
        # (B,T,C,H,W)
        #
        # Required:
        # (B,T,H,W,C)

        predictions = predictions.permute(
            0,
            1,
            3,
            4,
            2
        )

        return predictions

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x
    ):

        # Encoder
        latent = self.encode(
            x
        )

        # Last observed frame
        last_frame = x[:, -1]

        # Decoder
        prediction = self.decode(
            latent,
            last_frame
        )

        return prediction


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODEL A — TEMPORAL CONVLSTM SANITY TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    model = RainfallConvLSTM(
        input_channels=1,
        hidden_channels=64,
        output_channels=1,
        input_frames=12,
        output_frames=12
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------
    # Dummy input
    # --------------------------------------------------------

    x = torch.rand(
        2,
        12,
        20,
        30,
        1,
        device=device
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(x)

        features = model.encode(
            x
        )

    print(
        "Input shape      :",
        tuple(x.shape)
    )

    print(
        "Prediction shape :",
        tuple(prediction.shape)
    )

    print(
        "Feature shape    :",
        tuple(features.shape)
    )

    print(
        "Prediction min   :",
        prediction.min().item()
    )

    print(
        "Prediction max   :",
        prediction.max().item()
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert prediction.shape == (
        2,
        12,
        20,
        30,
        1
    )

    assert features.shape == (
        2,
        64,
        20,
        30
    )

    # log1p rainfall must be non-negative

    assert (
        prediction.min().item()
        >= 0
    )

    print()
    print(
        "STATUS: TEMPORAL CONVLSTM "
        "SANITY TEST PASSED"
    )

    print("=" * 70)