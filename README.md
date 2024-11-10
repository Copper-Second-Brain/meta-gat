# meta-gat

### Setting up project

#### Clone the Repo

```bash
git clone https://github.com/gamedevCloudy/meta-gat
cd meta-gat
```

#### Setup Virtual Environment

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Running the Model

#### CLI

```bash
python model.py
```

#### Running the Visualization

Running the api:

```bash
fastapi dev app
```

Runing the frontend:

Open the `app/index.html` using Live Server.

OR

```bash
cd app
python -m http.server 1432
```

Visit [https://localhost:1432](https://loclhost:1432/)
