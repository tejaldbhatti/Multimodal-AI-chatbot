import gradio as gr

SAMPLE_QUESTIONS = {
    "Budgeting": [
        ["What is the 50/30/20 rule for budgeting?"],
        ["Tell me about zero-based budgeting."]
    ],
    "Saving": [
        ["How much should I save monthly if I earn 4000 euros?"],
        ["What are some effective savings strategies?"]
    ]
}

def update_sample_questions_display(category_name):
    if category_name in SAMPLE_QUESTIONS:
        questions = SAMPLE_QUESTIONS[category_name]
        markdown_content = f"### Here are some sample questions for {category_name}:\n\n"
        for q_list in questions:
            markdown_content += f"- {q_list[0]}\n"
        markdown_content += "\nFeel free to ask your own questions as well!"
        return gr.update(value=markdown_content, visible=True), gr.update(value="", placeholder=f"Ask a question about {category_name}...")
    else:
        return gr.update(value="", visible=False), gr.update(value="", placeholder="Type your financial question here...")

# Dummy respond function, replace with your chatbot logic or API call
def respond(message, chat_history):
    bot_response = f"Chatbot response to: {message}"
    chat_history = chat_history or []
    chat_history.append((message, bot_response))
    return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("# 💰 Financial Literacy Chatbot Sample")

    with gr.Row():
        budgeting_btn = gr.Button("Budgeting")
        saving_btn = gr.Button("Saving")

    sample_questions_display = gr.Markdown(value="Select a category above to see sample questions.", visible=False)
    
    chatbot = gr.Chatbot()
    question_textbox = gr.Textbox(placeholder="Type your financial question here...")

    # Update sample questions when category buttons clicked
    budgeting_btn.click(fn=lambda: update_sample_questions_display("Budgeting"),
                        outputs=[sample_questions_display, question_textbox])
    saving_btn.click(fn=lambda: update_sample_questions_display("Saving"),
                     outputs=[sample_questions_display, question_textbox])

    # On submit in textbox, respond and update chatbot and clear input box
    question_textbox.submit(fn=respond,
                            inputs=[question_textbox, chatbot],
                            outputs=[question_textbox, chatbot])

demo.launch()
