FROM quay.io/astronomer/astro-runtime:3.2-3

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .