import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx: int):
        """idx에 해당하는 샘플을 데이터셋에서 불러오고 반환."""

        # 정답 레이블이 있다면 else문을, 없다면 if문을 수행
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    def __len__(self):
        """데이터셋 샘플 개수 반환."""

        return len(self.inputs)