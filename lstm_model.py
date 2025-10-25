import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=3, out_dim=256):
        """Light CNN for webcam/picam image encoding."""
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        # Instead of adaptive pooling → keep spatial tokens
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        """
        x -> [batch, channels, H, W]
        return -> [batch, num_patches, out_dim]
        """
        feat = self.conv_net(x)  # [B, 128, H/4, W/4]
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, patches, 128]
        return self.fc(feat)  # [B, patches, out_dim]


class CoordEncoder(nn.Module):
    def __init__(self, in_channels=4, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

    def forward(self, x):
        """x -> [batch, 4]"""
        return self.net(x)  # [batch, out_dim]



class CrossAttention(nn.Module):
    """ Cross Attention between visual feature and motor angle"""
    def __init__(self,visual_dim = 256, motor_dim = 128, hidden_dim = 128):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.motor_proj = nn.Linear(motor_dim , hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim ,num_heads= 4, batch_first = True)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 1, hidden_dim))  # Positional encoding for attention)
        ### STORE ATTENTION WEIGHT FOR VISUALIZATION
        self.attn_weights = None

    def forward(self,visual_feats, motor_feats):
        """visual_features -> [batch,seq_len, visual_dim ]"""
        """motor_features -> [batch,seq_len motor_dim]"""

        k = v  = self.visual_proj(visual_feats) + self.pos_encoding
        q = self.motor_proj(motor_feats) 

        attn_output, attn_weights = self.attn(q, k, v)
        self.last_attn_weights = attn_weights.detach().cpu()       

        return attn_output



class MotorAngleEmbedding(nn.Module):
    def __init__(self, motor_dim=3, embed_dim=32):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(motor_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, motor_angles):
        """ motor_angles -> [B, T, 3] """
        return self.embed(motor_angles)  # [B, T, embed_dim]


# class LSTMModel(nn.Module):
#     def __init__(self, hidden_dim=512, output_dim=3, motor_dim=3, num_layers=2, dropout=0.3, device='cuda'):
#         super().__init__()
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

#         self.spatial_encoder = SpatialEncoder(in_channels=3)
#         self.coord_encoder = CoordEncoder()
#         self.cross_attention = CrossAttention(visual_dim=256, motor_dim=32, hidden_dim=128)
#         self.motor_embed = MotorAngleEmbedding(motor_dim=motor_dim, embed_dim=32)

#         # LSTM input: visual (256) + cross-attn (128) + motor (32)
#         self.lstm = nn.LSTM(
#             input_size=416,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, picam_seq, webcam_seq, ball_coords, motor_seq):
#         """
#         picam_seq  -> [B, T, C, H, W]
#         webcam_seq -> [B, T, C, H, W]
#         ball_coords-> [B, T, 4]
#         motor_seq  -> [B, T, 3]
#         """

#         B, T = motor_seq.shape[:2]

#         visual_features = []
#         for t in range(T):
#             picam_feat = self.spatial_encoder(picam_seq[:, t])   # [B, patches, 256]
#             webcam_feat = self.spatial_encoder(webcam_seq[:, t]) # [B, patches, 256]
#             ball_feat = self.coord_encoder(ball_coords[:, t])    # [B, 128]

            
#             # ResNet (frozen)
#             with torch.no_grad():
#                 picam_res = self.resnet(picam_seq[:, t])
#                 webcam_res = self.resnet(webcam_seq[:, t])
#                 resnet_feat = (picam_res + webcam_res) / 2  # (batch, 512)

#             # Treat ball as an extra token
#             ball_feat = ball_feat.unsqueeze(1)                   # [B, 1, 128]
#             ball_feat = F.pad(ball_feat, (0, 128))               # match 256 dim
#             visual_t = torch.cat([picam_feat, webcam_feat, ball_feat], dim=1)  # [B, patches+patches+1, 256]
#             visual_features.append(visual_t)

#         visual_features = torch.stack(visual_features, dim=1)  # [B, T, num_tokens, 256]
#         motor_embeddings = self.motor_embed(motor_seq)         # [B, T, 32]

#         # Cross Attention → motor queries ball+vision
#         cross_attn = self.cross_attention(visual_features, motor_embeddings)  # [B, T, 128]

#         # LSTM input
#         lstm_input = torch.cat([visual_features.mean(dim=2), cross_attn, motor_embeddings], dim=-1)  # [B, T, 416]
#         lstm_out, _ = self.lstm(lstm_input)

#         return self.fc(lstm_out[:, -1, :])
class LSTMModel(nn.Module):
    def __init__(self,hidden_dim = 512,output_dim = 3, motor_dim = 3, num_layers = 4,dropout = 0.3, device= 'cuda'):
        super().__init__()
        self.device= torch.device(device if torch.cuda.is_available() else 'cpu')
        self.spatial_encoder = SpatialEncoder(in_channels = 3)
        self.coord_encoder = CoordEncoder()
        self.cross_attention = CrossAttention(visual_dim=256, motor_dim = 32)
        self.motor_embed = MotorAngleEmbedding(motor_dim=motor_dim,embed_dim = 32)

        #### Resnet pretrained model and frozen ####
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) 

        self.resnet.fc= nn.Identity()

        # vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        # self.vit = vit_b_16(weights=vit_weights)
        # self.vit.heads = nn.Identity()

        self.visual_proj = nn.Linear(512+ 128, 256)

        ##### LSTM Model #####
        self.lstm = nn.LSTM(
            input_size = 416,
            hidden_size = hidden_dim,
            num_layers=num_layers,
            batch_first= True,
            dropout= dropout if num_layers >1 else  0 ) #( 256  + 128 + 32)  

        ### output head 
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim , hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)

        )
    def forward(self, picam_seq,webcam_seq,ball_coords,motor_seq):
        """ picam_seq -> [batch, seq_len, channels, height, width] """
        """ webcam_seq -> [batch, seq_len, channels, height, width] """
        """ ball_coords -> [batch, seq_len, 4] """
        """ motor_seq -> [batch, seq_len, 3] """
    
        batch_size, seq_len = motor_seq.shape[:2]

        ### process each sequence ######
        visual_features= []
        for t in range(seq_len):
            picam_feat = self.spatial_encoder(picam_seq[:,t]) ## [batch, 256]
            webcam_feat = self.spatial_encoder(webcam_seq[:,t]) ## [batch, 256]
        

        ## process through resnet and take average features 
            with torch.no_grad():
                picam_vit = self.resnet(picam_seq[:,t])
                webcam_vit = self.resnet(webcam_seq[:,t])
                visual_feat_vit  = (webcam_vit + picam_vit)/2


            ball_feat = self.coord_encoder(ball_coords[:,t])

            visual_feat = torch.cat([ 
                visual_feat_vit,
                3.0 * ball_feat
            ], dim = 1)  ###(512 + 128)

            visual_feat = self.visual_proj(visual_feat) ## (batch, 256)
            visual_features.append(visual_feat)

        visual_features = torch.stack(visual_features, dim =1) ##(batch, seq_len , 256)
        motor_embeddings = self.motor_embed(motor_seq) ###(batch, seq_len , 32)

        cross_attn = self.cross_attention(visual_features, motor_embeddings) ## (batch, seq_len, 128)

        ### combine features for lstm ##

        lstm_input = torch.cat([visual_features,
                                cross_attn,
                                motor_embeddings], dim = -1)##(batch, seq_len , 416)
        
        lstm_out ,_ = self.lstm(lstm_input)
        return self.fc(lstm_out[:,-1,:])
