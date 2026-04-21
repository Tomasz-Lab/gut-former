import torch
from torch import nn

class BacteriaModel(nn.Module):
    def __init__(self, num_pathways, num_bacteria, embedding_dim, latent_size):
        super(BacteriaModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.latent_size = latent_size

        # Bacteria encoding
        self.pathways_embedding = nn.Embedding(num_pathways, embedding_dim)
        self.taxonomy_embedding = nn.Embedding(num_bacteria, embedding_dim)

        # Linear layer to transform (B, #Bac, 1) to (B, #Bac, D)
        self.linear_layer = nn.Linear(1, embedding_dim)

        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True, dropout=0.05)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)  # 1

        self.final_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=embedding_dim,
                                                              batch_first=True)
        self.final_transformer_encoder = nn.TransformerEncoder(self.final_encoder_layer, num_layers=1)

        # Mean and Logvar layers
        self.latent_layer = nn.Linear(embedding_dim, latent_size)

        # Linear transformation for latent vector
        self.linear_transform = nn.Linear(latent_size, embedding_dim)

        # Transformer decoder
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=1)

        # Final linear layer to transform (B, #Bac, D) to (B, #Bac, 1)
        self.latent_norm = nn.LayerNorm(latent_size)

        self.output_transform = nn.Sequential(
            nn.Linear(embedding_dim, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pathways_tensor, bacteria_tensor):

        batch_size = pathways_tensor.size(0)
        num_pathways = pathways_tensor.size(1)
        num_bacteria = bacteria_tensor.size(1)

        # Step 1: Transform (B, #Bac, 1) to (B, #Bac, D)
        pathways_transformed = self.linear_layer(pathways_tensor)
        bacteria_transformed = self.linear_layer(bacteria_tensor)

        # Step 2: Create bacteria encoding
        pathways_indices = torch.arange(num_pathways).to(pathways_tensor.device)
        pathways_encoded = self.pathways_embedding(pathways_indices)
        bacteria_indices = torch.arange(num_bacteria).to(bacteria_tensor.device)
        bacteria_encoded = self.taxonomy_embedding(bacteria_indices)

        # Expand bacteria encoding to match batch size and add to bacteria_transformed
        pathways_encoded_expanded = pathways_encoded.unsqueeze(0).expand(batch_size, -1, -1)
        pathways_result = pathways_transformed + pathways_encoded_expanded
        bacteria_encoded_expanded = bacteria_encoded.unsqueeze(0).expand(batch_size, -1, -1)
        bacteria_result = bacteria_transformed + bacteria_encoded_expanded

        res_concat = torch.cat([bacteria_result, pathways_result], dim=1)

        # Step 3: Pass through Transformer encoder
        output = self.transformer_encoder(res_concat)
        output = self.final_transformer_encoder(output)

        # Step 4: Average over sequence (mean along the sequence dimension)
        mean_vector = output.mean(dim=1)

        # Step 5: Compute mean and logvar
        latent = self.latent_layer(mean_vector)

        # Step 7: Transform z to (B, D)
        D_vector = self.linear_transform(latent)

        # Step 8: Repeat D_vector for each bacterium to match (B, #Bac, D)
        D_vector_expanded = D_vector.unsqueeze(1).repeat(1, num_pathways, 1)
        pathways_result_with_encoding = D_vector_expanded + pathways_encoded_expanded

        D_vector_expanded_taxonomy = D_vector.unsqueeze(1).repeat(1, num_bacteria, 1)
        bacteria_result_with_encoding = D_vector_expanded_taxonomy + bacteria_encoded_expanded

        # Step 9: Pass through Transformer decoder
        pathways_output = self.transformer_decoder(pathways_result_with_encoding)  # zamienic na encoder i usunąć
        bacteria_output = self.transformer_decoder(bacteria_result_with_encoding)  # zamienic na encoder i usunąć

        # Step 10: Transform output to (B, #Bac, 1)
        pathways_output_transformed = self.output_transform(pathways_output)
        bacteria_output_transformed = self.output_transform(bacteria_output)

        return bacteria_output_transformed, pathways_output_transformed, latent