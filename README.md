# Multi-Agent Virtual Company 🤖

A sophisticated multi-agent research system built with LangGraph that orchestrates specialized AI agents to perform comprehensive research, analysis, and report generation on stocks and tech trends.

## 🌟 Project Overview

This project demonstrates advanced multi-agent orchestration where a **Supervisor Agent** coordinates multiple specialized agents to complete complex research tasks. Each agent has a specific role:

- **Supervisor**: Orchestrates the workflow and decides which agent should act next
- **Researcher**: Gathers data from the web using Tavily API
- **Analyst**: Processes and summarizes raw research data
- **Critic**: Reviews outputs for quality and provides feedback
- **Writer**: Produces polished final reports

## 🏗️ Architecture

```
    ┌─────────────┐
    │  Supervisor │  ← Orchestrates workflow
    └──────┬──────┘
           │
    ┌──────┴──────┬──────────┬─────────┐
    ▼             ▼          ▼         ▼
┌────────┐  ┌─────────┐  ┌──────┐  ┌────────┐
│Research│  │ Analyst │  │Critic│  │ Writer │
└────────┘  └─────────┘  └──────┘  └────────┘
```

## 📁 Project Structure

```
Multi Agent Researcher/
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration management
├── src/
│   ├── main.py              # CLI entry point
│   ├── app.py               # Streamlit UI (upcoming)
│   ├── agents/              # Agent implementations
│   ├── graph/               # LangGraph workflow
│   ├── tools/               # Search and analysis tools
│   ├── prompts/             # Agent prompts
│   └── schemas/             # Pydantic models
├── tests/                   # Test suite
├── outputs/                 # Generated reports
├── requirements.txt
├── .env.example
└── README.md
```

## 🛠️ Tech Stack

- **LangGraph**: Multi-agent orchestration framework
- **Groq**: Fast LLM inference (llama-3.3-70b-versatile)
- **Tavily API**: Web search optimized for AI agents
- **Pydantic**: Data validation and type safety
- **Streamlit**: Interactive UI (upcoming)
- **Python 3.13**: Core language

## 📝 Usage

*Coming soon - implementation in progress*

```python
# CLI usage
python src/main.py --topic "Tesla stock analysis"


## 📄 License

This project is for educational and portfolio purposes.

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

---

**Built with ❤️ for AI portfolio showcase**
