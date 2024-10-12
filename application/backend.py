import gradio

with open('css/style.css', 'r') as file:
    css = file.read()
  
with open('js/scripts.js', 'r') as file:
    js = file.read()  
  

head = f"""
<script>
    {js}
</script> 
"""  
    
embeddings = ["emb1", "emb2", "emb3"]
models = ["resnet50", "densenet161", "vgg16"]

def print_func(save_xformers, save_path, emb_name, use_emb, scheduler_dd):
    print(save_xformers)
    print(save_path)
    print(emb_name)
    print(use_emb)
    print(scheduler_dd)
    
    
with gradio.Blocks(css=css, head=head) as app:
    gradio.HTML(
         f"""
            <div class="diffusion-spave-div">
              <div>
                <h1>Diffusion Space</h1>
              </div>
            </div>
        """
    )
    with gradio.Row():
        with gradio.Column(scale=15):
            with gradio.Tab("Prompting"):
                generate = gradio.Button(value="Documentation", variant="secondary", elem_classes="custom-button",
                                         link="https://sygil-dev.github.io/sygil-webui/docs/Installation/docker-guide/")
                
        with gradio.Column(scale=55):
            with gradio.Group():
                gallery = gradio.Gallery(
                    label="Generated image",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    container=True,
                    interactive=True,
                    show_share_button=True,
                    show_download_button=True,
                    object_fit="fill",
                    min_width=600,
                )
                
            settings = gradio.Markdown()
            error_output = gradio.Markdown()
            
        with gradio.Column(scale=30):
            
            with gradio.Tab("Controllers"):
                
                generate = gradio.Button(value="Generate", variant="secondary")
            
                gradio.HTML(
                    f"""
                        <div class="normal-text">
                        <div>
                            <p>Try to prompt in the most detailed way for the deep learning model
                            to produce the best result. Feel free to try as many prompt as you can.</p>
                        </div>
                        </div>
                    """
                )
            
                with gradio.Group():
                    prompt = gradio.Textbox(label="Prompt", show_label=False, 
                                            max_lines=3, placeholder="Enter prompting text", 
                                            lines=10, container=False, elem_id="prompt-space")
                    
                    neg_prompt = gradio.Textbox(label="Negative prompt", show_label=True, 
                                                placeholder="What to exclude from the image")
                    
                    with gradio.Row():
                        n_images = gradio.Slider(label="Images", interactive=True, value=5, minimum=0, maximum=12, step=1, elem_classes="custom-slider")
                        seed = gradio.Slider(label='Seed', interactive=True, value=10000, minimum=1000, maximum=2147483647, step=1, elem_classes="custom-slider")

                    with gradio.Row():
                        guidance = gradio.Slider(label="Guidance scale", interactive=True, value=4, minimum=0, maximum=20, step=1, elem_classes="custom-slider")
                        steps = gradio.Slider(label="Steps", interactive=True, value=10, minimum=10, maximum=50, step=1, elem_classes="custom-slider")

                    with gradio.Row():
                        width = gradio.Slider(label="Width", interactive=True, value=100, minimum=64, maximum=1920, step=64, elem_classes="custom-slider")
                        height = gradio.Slider(label="Height", interactive=True, value=570, minimum=64, maximum=1920, step=64, elem_classes="custom-slider")

                    scheduler_dd = gradio.Radio(
                        label="Scheduler",
                        value="euler_a",
                        type="value",
                        interactive=True,
                        choices=["euler_a", "euler", "dpm++", "ddim", "ddpm", "pndm", "lms", "heun", "dpm"],
                        elem_classes="custom-radio-button" 
                    )
                    
                with gradio.Group(): 
                    with gradio.Row():  
                        use_emb = gradio.Checkbox(label="Use Embedding", value=False, container=True, elem_classes="custom-checkbox")
                        emb_name = gradio.Dropdown(label="Emb Name", 
                                                choices=[e for e in embeddings], 
                                                value=embeddings[0], 
                                                interactive=True, 
                                                show_label=False)
                    with gradio.Row():
                        emb_weight = gradio.Slider(label="EMB Weight", interactive=True, minimum=0, maximum=1, step=0.05, value=1, elem_classes="custom-slider")
                        te_weight = gradio.Slider(label="TE Weight", interactive=True, minimum=0, maximum=1, step=0.05, value=1, elem_classes="custom-slider")
                
            
            with gradio.Tab("Image Conversion"):
                with gradio.Group():
                    image = gradio.Image(label="Image", height=256, type="pil")
                    strength = gradio.Slider(label="Transformation strength", minimum=0, maximum=1, step=0.01, value=0.5, interactive=True, elem_classes="custom-slider")
                    generate2 = gradio.Button(value="Generate", variant="secondary")

            with gradio.Tab("Tools"):
                with gradio.Group():
                    pngInfo = gradio.Image(label="PNG Chunk Explorer", height=256, type="pil", elem_classes="custom-image-uploader")
                    pngText = gradio.Markdown()

                with gradio.Group():
                    txt = gradio.Textbox(lines=1, label="Initial Text", placeholder="Prompt start here", show_label=False)
                    out = gradio.Textbox(lines=4, label="Generated Prompts")
                    
                with gradio.Row():
                    gen = gradio.Button(value="Make Prompt", elem_classes="custom-button")
                    send_prompt = gradio.Button(value="Send to Prompt", elem_classes="custom-button")
                        
                gradio.HTML("""
                    <div style="border-top: 1px solid #303030;">
                        <br>
                        <p>Using MagicPrompt by <a href="https://huggingface.co/Gustavosta">Gustavosta</a> <3</p>
                    </div>
                """)
                
            with gradio.Tab("Settings"):
                refresh = gradio.Button(value="Refresh Models")
                settings_message = gradio.Markdown()
                
                with gradio.Group():
                    model_name = gradio.Dropdown(label="Model", choices=[m for m in models], value=models[0], interactive=True)
                    save_vmodel = gradio.Checkbox(label="V-Model", value=True, elem_classes="custom-checkbox", interactive=True)
                    save_xformers = gradio.Checkbox(label="Xformers", value="use_xformers", elem_classes="custom-checkbox", interactive=True)

                with gradio.Group():
                    save_path = gradio.Textbox(label="Image save Path", value="saving_path", interactive=True)  
                    
                save_path_button= gradio.Button(value="Save Path", elem_classes="custom-button-2")
                
                
    save_path_button.click(
        print_func,
        inputs=[save_xformers, save_path, emb_name, use_emb, scheduler_dd],
        outputs=[]
    )
                
    

app.launch(debug=True)