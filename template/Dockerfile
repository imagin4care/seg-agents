# The image of your agent: Python, the kit, your code and its manifest.
# Add what your model needs (pip install ..., the weights) above the COPY of
# your own files, so that a change of your code does not install it all again.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /agent
COPY seg_agent_kit.py agent.py manifest.json ./
EXPOSE 8000
CMD ["python", "agent.py"]
