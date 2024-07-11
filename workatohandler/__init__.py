import logging
import azure.functions as func
from FlaskApp import app

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Received HTTP request, processing with Flask application.')

    try:
        response = func.WsgiMiddleware(app.wsgi_app).handle(req, context)
        logging.info('Request processed successfully.')
        return response
    except Exception as e:
        logging.error(f'Error processing request: {str(e)}', exc_info=True)
        return func.HttpResponse("Internal Server Error", status_code=500)
