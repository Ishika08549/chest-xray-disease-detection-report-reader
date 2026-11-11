# Chest X-Ray Disease Detection & Radiology Report Reader

A React-based dashboard for an educational deep-learning project that combines chest X-ray classification, Grad-CAM visualization, and radiology report retrieval.

## Features

- Upload chest X-ray images in PNG or JPEG format.
- Display predictions for Normal, Opacity, Cardiomegaly, and Atelectasis.
- Visualize model attention using Grad-CAM.
- Retrieve verified radiology reports using dataset image IDs.
- Display backend connection status and error messages.

## Technology Stack

- React
- Vite
- JavaScript
- CSS
- FastAPI backend

## Requirements

- Node.js and npm
- Python environment with backend dependencies installed

## Run the Frontend

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, usually:

http://localhost:5173

## Run the Backend

Open a separate terminal from the project root:

```bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --reload
```

Backend API documentation:

http://127.0.0.1:8000/docs

The frontend communicates with the local backend at:

http://127.0.0.1:8000

## Production Build

```bash
npm run lint
npm run build
```

## Important Disclaimer

This project is intended for educational and research purposes only. Model predictions and Grad-CAM visualizations are experimental outputs and must not be used for medical diagnosis or clinical decision-making. Retrieved radiology reports are original dataset records, not newly generated medical advice.
