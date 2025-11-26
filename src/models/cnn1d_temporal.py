import torch
import torch.nn as nn


class CNN1DTemporal(nn.Module):
    """
    Simple 1D convolutional temporal predictor.

    Input  : (batch, 2, Nx)   -> [rho_norm(t_k, ·), p_norm(t_k, ·)]
    Output : (batch, 2, Nx)   -> predicted [rho_norm(t_{k+1}, ·), p_norm(t_{k+1}, ·)]
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2  # 'Same' padding (assuming odd kernel_size)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels,
                      kernel_size = kernel_size, padding = padding),
            nn.ReLU(inplace = True),

            nn.Conv1d(hidden_channels, hidden_channels,
                      kernel_size = kernel_size, padding = padding),
            nn.ReLU(inplace = True),

            nn.Conv1d(hidden_channels, in_channels,
                      kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 2, Nx)
        return self.net(x)
