import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))

        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)

        alpha = F.softmax(attention_scores, dim=1)

        attention_weights = features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)

        return alpha, attention_weights


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob=0.3):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        # self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.gru = nn.GRUCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.device = device

    def init_hidden(self, enc_out):
        mean_encoder_out = enc_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)

        return h

    def forward(self, features, captions):
        embeds = self.embedding(captions)
        hidden_state = self.init_hidden(features)

        batch_size = captions.size(0)

        seq_lengths = len(captions[0]) - 1
        features_size = features.size(1)

        preds = torch.zeros(batch_size, seq_lengths, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_lengths, features_size).to(self.device)

        for s in range(seq_lengths):
            alpha, context = self.attention(features, hidden_state)
            gru_input = torch.cat((embeds[:, s], context), dim=1)

            hidden_state = self.gru(gru_input, hidden_state)

            output = self.fcn(self.drop(hidden_state))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def predict(self, features, max_len=20):
        batch_size = features.size(0)
        hidden_state = self.init_hidden(features)

        alphas = []
        word = torch.tensor(1).view(1, -1).to(self.device)  # index of \s token is 1
        embeds = self.embedding(word)
        captions = []

        for i in range(max_len):
            alpha, context = self.attention(features, hidden_state)
            gru_input = torch.cat((embeds[:, 0], context), dim=1)

            hidden_state = self.gru(gru_input, hidden_state)
            output = self.fcn(self.drop(hidden_state))
            output = output.view(batch_size, -1)

            predicted_word_idx = output.argmax(dim=1)

            # save the generated word
            captions.append(predicted_word_idx.item())

            # end if <EOS detected>
            if predicted_word_idx.item() == 2:
                break

            # send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

            # covert the vocab idx to words and return sentence
        return captions


class ImageCaption(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, device, drop_prob=0.3):
        super(ImageCaption, self).__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            drop_prob=drop_prob,
            device=device
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def predict(self, images, max_len=20):
        features = self.encoder(images)
        outputs = self.decoder.predict(features, max_len)
        return outputs
