# OpenAI Agents SDK Hackathon Boilerplate

This project is a robust starter template for building AI agent apps using the OpenAI Agents SDK combined with Streamlit for a polished chat interface.  
Designed for hackathons and rapid prototyping, it includes multi-agent handoffs, input validation guardrails, fallback logic, and a clean, user-friendly UI.

---

## Features

- Multi-agent system with delegation via handoffs based on keywords
- Input validation using guardrails for profanity, input length, and sensitive info detection
- Built-in fallback agent for offline or API failure scenarios
- Streamlit-based chat UI with styled user and agent bubbles, scrollable history, and debug trace toggle
- Easy to extend with custom tools and agents

---

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (get yours from [OpenAI](https://platform.openai.com/account/api-keys))

### Installation

1. Clone this repository or download the files.

2. Create a `.env` file in the root directory and add your OpenAI API key:

```

OPENAI\_API\_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

````

3. Install the required Python packages:

```bash
pip install -r requirements.txt
````

---

### Running the Application

You can run the Streamlit app using either of the following commands:

```bash
streamlit run app.py
```

If the above command does not work (sometimes due to PATH issues), run:

```bash
python -m streamlit run app.py
```

---

## Usage

* Type your question or command in the input box.
* The app will route your request to the appropriate agent (Calculator or Search) based on keywords.
* Input validation will block messages with profanity, excessive length, or sensitive info.
* If OpenAI API calls fail, a fallback agent will respond gracefully.
* Use the **Show raw agent trace** checkbox to debug the internal agent flow.

---

## Extending the Boilerplate

* Add new tools by defining Python functions and wrapping them with `Tool.from_function`.
* Create more specialized agents and define handoffs for delegation.
* Enhance guardrails with additional checks as needed.
* Customize the Streamlit UI in `app.py` for your branding or features.

---

## Troubleshooting

* Make sure your OpenAI API key is correct and active.
* Ensure all dependencies are installed (`pip install -r requirements.txt`).
* If you get errors running Streamlit, try using `python -m streamlit run app.py`.
* For local testing without API access, the fallback agent will handle requests but with limited capability.

---

## License

MIT License © 2025 Muhammad Bin Zohaib

---

## Contact

Feel free to reach out for collaborations or questions:
**Email:** [muhammad.specials@gmail.com](mailto:muhammad.specials@gmail.com)
**LinkedIn:** [mbz1415](https://www.linkedin.com/in/mbz1415/)

---

Happy hacking! 🚀