#!/bin/bash

addrport="0.0.0.0:8000"

# Wait for backend connection
/bin/bash wait-for-it.sh

## Needs a connection to a DB so migrations go here
python manage.py makemigrations website
python manage.py migrate
python manage.py createuser admin $ADMIN_PASSWORD --superuser
python manage.py stopcelery 
python manage.py startcelery 

echo ""
echo "-=------------------------------------------------------"
echo " Starting the web server on '$addrport'..."
echo "-=------------------------------------------------------"
python manage.py runserver "$addrport"
