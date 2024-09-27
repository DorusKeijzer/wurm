# Prepare your optimizer to only update unfrozen parameters
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005,
    momentum=0.9
)

# Example training loop (simplified)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    # Your training code here
    # For example:
    #   1. Load your data
    #   2. Forward pass through the model
    #   3. Compute loss
    #   4. Backpropagation and optimization step

# Save the model after training
torch.save(model.state_dict(), "faster_rcnn_medical_imaging.pth")
