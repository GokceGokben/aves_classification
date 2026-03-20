import torch  # pyre-ignore[21]
import torch.nn as nn  # pyre-ignore[21]
import torchvision.models as models  # pyre-ignore[21]

class MultiHeadBirdModel(nn.Module):
    def __init__(self, num_species=400):
        super(MultiHeadBirdModel, self).__init__()
        
        # 1. Backbone: EfficientNet-B0 is used
        # weights='DEFAULT' loads the weights trained on ImageNet
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        
        # Get the number of input features of the last classifier layer of EfficientNet
        # Usually 1280.
        num_features = self.backbone.classifier[1].in_features
        
        # Remove the original classifier (disable it by making it Identity)
        self.backbone.classifier = nn.Identity()
        
        # 2. Heads: Separate linear layer for each task
        # Species Prediction (400+ classes)
        self.species_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_species)
        )
        

    def forward(self, x):
        # Extract features (Feature Extraction)
        features = self.backbone(x)
        
        # Each head makes its own prediction
        species_out = self.species_head(features)
        
        return species_out

if __name__ == "__main__":
    # 1. Model creation (400 species prediction)
    model = MultiHeadBirdModel(num_species=400)
    
    # 2. Create a dummy image (Batch: 32, Channel: 3, Height: 224, Width: 224)
    dummy_input = torch.randn(32, 3, 224, 224)
    
    # 3. Test the model
    print("Model test is starting...")
    output = model(dummy_input)  # pyre-ignore
    
    # 4. Check output dimensions
    print(f"Species output form: {output.shape}") # [32, 400]
    print("Test completed successfully! Layers are connected correctly.")