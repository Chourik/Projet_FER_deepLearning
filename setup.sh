mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml


7. heroku login
   heroku create
   git add .
   git commit -m "Some message"
   git push heroku master