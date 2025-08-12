#!/bin/bash
concurrently \
  "cd /app && python3 app.py" \
  "cd /app/FrontEnd && npm start"
