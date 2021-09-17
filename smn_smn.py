import torch
from torch import nn
import math
from torch.nn import init
from torch.nn import functional as F

'''

class SMN(nn.Module):

    def __init__(self,voc_len):
        super(SMN, self).__init__()
        VOCAB_LEN = voc_len
        max_sent = 50
        cnn_out_channels = 8
        cnn_kernel_size = 3
        cnn_padding = 0
        cnn_out_dim = cnn_out_channels * ((max_sent + 2 * cnn_padding - (cnn_kernel_size - 1)) // cnn_kernel_size) ** 2
        match_dim = 50
        self.emb_dim = 200

        self.emb = nn.Embedding(VOCAB_LEN, 200, padding_idx=0)

        self.gru_1 = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)
        self.weight = nn.Parameter(torch.randn(self.emb_dim, self.emb_dim), requires_grad=True)
        bound = 1 / math.sqrt(self.emb_dim)
        init.uniform_(self.weight, -bound, bound)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_padding),
            nn.ReLU(),
            nn.MaxPool2d(cnn_kernel_size)
        )
        self.cnn2match = nn.Linear(cnn_out_dim, match_dim)
        self.gru_2 = nn.GRU(match_dim, match_dim, batch_first=True)
        self.final = nn.Linear(match_dim, 1)

    def forward(self, cs, r):
          # BSL, BL

        mask2d, _, _ = get_masks(cs, r)  ## BSNM
        mask2d = mask2d.permute(0, 1, 3, 2)  # BSMN

        mask_sequence = (cs == 0).all(-1)  # BS
        len_sequence = mask_sequence.bitwise_not().long().sum(-1)  # BS

        r = self.emb(r)
        rg = self.gru_1(r)[0]

        cs = self.emb(cs)  # BSLE
        cs_flat = cs.view(-1, *cs.shape[2:])  # (B*S)ME
        cg = self.gru_1(cs_flat)[0].view(*cs.shape)  # BSMH

        m1 = torch.einsum('bsme,bne->bsmn', [cs, r])
        m2 = torch.einsum('bsme,ef,bnf->bsmn', [cg, self.weight, rg])
        m = torch.stack([m1, m2], 2)  # BS2MN
        m = m.masked_fill(mask2d.unsqueeze(2), 0)

        m = self.cnn(m.view(-1, *m.shape[2:])).view(*m.shape[:2], -1)

        m = self.cnn2match(m)

        m = m.masked_fill(mask_sequence.unsqueeze(-1), 0)

        # m = pack_padded_sequence(m, len_sequence, batch_first=True, enforce_sorted=False)
        msg = self.gru_2(m)[0]
        # msg = pad_packed_sequence(msg, batch_first=True, total_length=cs.shape[1])[0]

        # msg = msg.masked_fill(mask_sequence.unsqueeze(-1), 0)
        msg = msg[torch.arange(0, msg.size(0)), len_sequence - 1]
        score = self.final(msg).squeeze(-1)

        return torch.sigmoid(score).view(-1,1)



'''
class SMN(nn.Module):

    def __init__(self,voc_len):
        super(SMN, self).__init__()
        VOCAB_LEN = voc_len
        max_sent = 50
        cnn_out_channels = 8
        cnn_kernel_size = 3
        cnn_padding = 0
        cnn_out_dim = cnn_out_channels * ((max_sent + 2 * cnn_padding - (cnn_kernel_size - 1)) // cnn_kernel_size) ** 2
        match_dim = 50
        self.emb_dim = 200

        self.emb = nn.Embedding(VOCAB_LEN, 200, padding_idx=0)

        self.gru_1 = nn.GRU(self.emb_dim, self.emb_dim, batch_first=True)

        self.weight = nn.Parameter(torch.zeros(self.emb_dim, self.emb_dim), requires_grad=True)
        bound = 1 / math.sqrt(self.emb_dim)
        init.uniform_(self.weight, -bound, bound)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding=cnn_padding),
            nn.ReLU(),
            nn.MaxPool2d(cnn_kernel_size)
        )
        self.cnn2match = nn.Linear(cnn_out_dim, match_dim)
        self.gru_2 = nn.GRU(match_dim, match_dim, batch_first=True)
        self.final = nn.Linear(match_dim, 1)

    def forward(self, cs, r):
        #cs, r = x  # BSL, BL

        mask2d, _, _ = get_masks(cs, r)  ## BSNM
        mask2d = mask2d.permute(1, 0, 3, 2)  # SBMN
        #
        mask_sequence = (cs == 0).all(-1)  # BS

        r = self.emb(r)
        rg = self.gru_1(r)[0]

        ms = []  # TBE
        for c, mask in zip(cs.transpose(0, 1), mask2d):
            # for c in cs.transpose(0, 1):
            c = self.emb(c)
            cg = self.gru_1(c)[0]

            m1 = torch.einsum('bme,bne->bmn', [c, r])
            m2 = torch.einsum('bme,ee,bne->bmn', [cg, self.weight, rg])

            m = torch.stack([m1, m2], 1)  # 2BMN
            m = m.masked_fill(mask.unsqueeze(1), 0)

            m = self.cnn(m)

            m = m.view(m.shape[0], -1)  # (MAX_SENT - 2) // 3

            m = F.relu(self.cnn2match(m))

            ms.append(m)

        ms = torch.stack(ms).transpose(0, 1)  # BSE
        ms.masked_fill(mask_sequence.unsqueeze(-1), 0)
        msg = self.gru_2(ms)[0]

        score = self.final(msg[:, -1, :]).squeeze(-1)
        return torch.sigmoid(score).view(-1,1)


def get_masks(cs, r):
    mask2d_r_cs = (cs == 0).unsqueeze(-2).repeat(1, 1, 50, 1).__or__(
        (r == 0).unsqueeze(1).unsqueeze(-1).repeat(1, 10, 1, 50))  # BSNM
    mask2d_r_r = (r == 0).unsqueeze(-2).repeat(1, 50, 1).__or__(
        (r == 0).unsqueeze(-1).repeat(1, 1, 50))  # BNN
    mask2d_cs_cs = (cs == 0).unsqueeze(-2).repeat(1, 1, 50, 1).__or__(
        (cs == 0).unsqueeze(-1).repeat(1, 1, 1, 50))  # BSMM

    return mask2d_r_cs, mask2d_cs_cs, mask2d_r_r

