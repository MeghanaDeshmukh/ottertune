#!/bin/bash

# Wait for MySQL connection
/bin/bash wait-for-it.sh

## Needs a connection to a DB so migrations go here
python manage.py makemigrations website
python manage.py migrate

python manage.py stopcelery
python manage.py startcelery
python manage.py runserver 0.0.0.0:8000 &

# Wait for server
sleep 10

# Integration tests
cd /app/client/driver
fab integration_tests
