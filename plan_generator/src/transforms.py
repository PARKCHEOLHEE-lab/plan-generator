from typing import Tuple
import torch


class TransformMirroring:
    def __call__(self, data: Tuple[torch.Tensor], mirroring_dimension: int) -> Tuple[torch.Tensor]:
        """Apply the mirroring transform to be vertical or horizontal by the `mirroring_dimension`
        The dimension for mirroring the data is determined by local random

        Args:
            data (torch.Tensor): data to mirror
            mirroring_dimension (int): axis to mirror

        Returns:
            torch.Tensor: mirrored data
        """

        mirrored = torch.vstack(data).flip(mirroring_dimension)

        return mirrored[0].unsqueeze(0), mirrored[1].unsqueeze(0), mirrored[2].unsqueeze(0)


class TransformRotating:
    def __call__(self, data: Tuple[torch.Tensor], rotation_multiplier: int) -> Tuple[torch.Tensor]:
        """Apply the rotating transform by the `rotation_multiplier`

        Args:
            data (Tuple[torch.Tensor]): data to rotate
            rotation_multiplier (int): rotate by 90 x this value

        Returns:
            Tuple[torch.Tensor]: rotated data
        """

        rotated = torch.rot90(torch.vstack(data), rotation_multiplier, dims=(1, 2))

        return rotated[0].unsqueeze(0), rotated[1].unsqueeze(0), rotated[2].unsqueeze(0)
