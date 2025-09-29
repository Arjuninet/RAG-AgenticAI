import gradio as gr

def sentence_builder(quantity, tech_worker_type, countries, place, activity_list, morning):
    return f"""The {quantity} {tech_worker_type}s from {" and ".join(countries)} 
    went to the {place} where they {" and ".join(activity_list)} until 
    the {"morning" if morning else "night"}"""

demo  = gr.Interface(
    fn=sentence_builder,
    inputs=[
        gr.Slider(3,20, value=4, step=1, label="Count",info="Choose between 3 and 20"),
        gr.Dropdown(
            ["Data Scientist","Data Analyst","ML Engineer"],
            label="tech worker type",
            info="Will add more tech worker types later!"
        ),
        gr.CheckboxGroup(["Canada","Japan","India"], label="Countries", info="where are they from?"),
        gr.Radio(["office","resturant","meeintg room"], label="Location", info="where did they go?"),
        gr.Dropdown(
            ["partied", "brainstormed", "coded", "fixed bugs"], 
            value=["brainstormed", "fixed bugs"], 
            multiselect=True, 
            label="Activities", 
            info="Which activities did they perform?"
        ),
        gr.Checkbox(label="Morning", info="Did they do it in the morning"),
    ],
    outputs="text",
    examples=[
        [3, "Software Developer", ["Canada", "Japan"], "restaurant", ["coded", "fixed bugs"], True],
        [4, "Data Scientist", ["Japan"], "office", ["brainstormed", "partied"], False],
        [10, "Software Engineer", ["Canada", "France"], "meeting room", ["brainstormed"], False],
        [8, "Data Scientist", ["France"], "restaurant", ["coded"], True],
    ]
)

demo.launch(server_name="127.0.0.1", server_port=7860, share=True)