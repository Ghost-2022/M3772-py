import click
from flask import render_template, jsonify
from app.scripts.setup import init_db

from app import create_app

web_app = create_app()


@web_app.route('/')
def index():
    return render_template('index.html')


@web_app.errorhandler(500)
def service_error(error):
    return {'code': 500}


@web_app.errorhandler(404)
def page_not_found(error):
    return {'code': 404}


@click.command()
@click.option('--func',
              help='')
def main(func):
    if func == 'init_db':
        init_db()
    elif func == 'run':
        web_app.run(host='0.0.0.0', port=8000)


if __name__ == '__main__':
    main()