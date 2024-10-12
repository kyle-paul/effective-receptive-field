import torch
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image

with open('css/style.css', 'r') as file:
    css = file.read()
  
with open('js/scripts.js', 'r') as file:
    js = file.read()  
  

head = f"""
<script>
    {js}
</script> 
"""  

class ERF:
    def __init__(self):
        self.model = None
        self.list_models = timm.list_models()
        
        with gr.Blocks(css=css, head=head) as self.app:
            with gr.Tab("Effective Receptive Field"):
                self.read_labels()
                self.tab1 = gr.Interface(
                    title="Effective Receptive Field",
                    fn=self.deep_visualization,
                    inputs = [
                        gr.Image(type="pil"),
                        gr.Slider(minimum=0, maximum=1, value=0.15, label="Positive Thresholding (%)"),
                        gr.Slider(minimum=0, maximum=1, value=0.15, label="Negative Thresholding (%)"),
                        gr.Slider(minimum=0, maximum=1, value=0.6, label="Opacity Controller"),
                        gr.Dropdown(choices=self.list_models, value="resnet50", label="Select models")
                    ],
                    outputs=[
                        gr.Image(label="Receptive Field Analysis"),
                        gr.Label(num_top_classes=10, label="Classification"),
                    ],
                    examples=[
                        ["/workspace/samples/cock.jpg", 0.15, 0.15, 0.6, "resnet50"],
                        ["/workspace/samples/cats.jpg", 0.15, 0.15, 0.6, "vgg16"],
                        ["/workspace/samples/goldfish.jpeg", 0.15, 0.15, 0.6, "densenet161"]
                    ],
                )
                
            with gr.Tab("Deep Visualization"):
                gr.Markdown("Deep Visualization")
        
    
    def read_labels(self):
        with open("assets/imagenet-1k.txt", 'r') as file:
            lines = file.readlines()
            self.labels = [line.strip().strip('"')[:-2] for line in lines]
    
    def initialize_model(self):
        self.model = timm.create_model(self.option, pretrained=True).cuda()
        
    def preprocess(self):
        preprocess_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.transform = timm.data.create_transform(**preprocess_cfg)
        self.model.eval()
        

    def deep_visualization(self, image, pos_thres, neg_thres, opacity, option):
        self.image = image
        self.opacity = opacity
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.option = option
        
        # Intialize model and preprocessor
        self.initialize_model()
        self.preprocess()
        
        # Forward pass
        x = self.transform(image).unsqueeze(0).requires_grad_(True).cuda()
        x.retain_grad()
        y = self.model(x)
        z = self.model.forward_features(x)
        
        # Results
        probs = F.softmax(y, dim=1).flatten()
        confidences = {self.labels[i]: float(probs[i]) for i in range(1000)}
        
        # Effective receptive field
        self.eff_recep_field(x, z)
        eff_recep_field = Image.open(".cache/eff_recep_field.png")
        
        return eff_recep_field, confidences
                
    
    def eff_recep_field(self, x, z):
        loss_vec = z[0, :, z.size(-2)//2, z.size(-1)//2]
        loss = torch.sum(loss_vec)
        loss.backward()
        
        # take the mean gradient between channels
        gradients = torch.mean(x.grad[0], dim=0).detach().cpu().numpy()

        # seperate pos/neg to avoid imbalance
        pos_grads = np.where(gradients >= 0, gradients, 0)
        neg_grads = np.where(gradients < 0, gradients, 0)
        neg_grads = np.abs(neg_grads)

        # Threhold -> binary
        pos_grads = np.where(pos_grads >= np.max(pos_grads) * self.pos_thres, pos_grads, 0)
        neg_grads = np.where(neg_grads >= np.max(neg_grads) * self.neg_thres, neg_grads, 0)
        
        plt.imshow(np.array(self.image.resize((224, 224))))
        plt.imshow(pos_grads, alpha=self.opacity, cmap="nipy_spectral")
        plt.imshow(neg_grads, alpha=self.opacity, cmap="nipy_spectral")
        
        plt.axis('off')
        plt.savefig('.cache/eff_recep_field.png', bbox_inches='tight', pad_inches=0)
        
    
    def run(self):
        self.app.launch(debug=True)
        

    
if __name__ == "__main__":
    erf = ERF()
    erf.run()