# Lane Visualization Web Application

This application provides a web-based interface for visualizing and editing lane data. It consists of a Python Flask backend and a React frontend.

## Setup

### Backend

1.  **Navigate to the backend directory:**
    ```bash
    cd web/backend
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Frontend

1.  **Navigate to the frontend directory:**
    ```bash
    cd web/frontend
    ```

2.  **Install the required Node.js packages:**
    ```bash
    npm install
    ```

## Running the Application

1.  **Start the backend server:**
    From the `web/backend` directory, run:
    ```bash
    python app.py
    ```
    The backend server will start on port 5000.

2.  **Start the frontend development server:**
    From the `web/frontend` directory, run:
    ```bash
    npm start
    ```
    The frontend development server will start on port 3000, and the application will open in your default web browser.

## Data

The application loads lane data from the `lanes/TEMP` directory. Make sure this directory exists and contains the necessary `.npy` files.
