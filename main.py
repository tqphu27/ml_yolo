from multiprocessing import Process
import os
import argparse
import logging
import logging.config

from flask_cors import CORS
from aixblock_ml.api import init_app
from model import MyModel
from flask import Flask, jsonify, send_from_directory, send_file, request, abort
import json

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return dict()
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config

def tensorboard_main(host, port, logdir):
    configuration = list([""])
    configuration.extend(["--host", host])
    configuration.extend(["--port", port])
    configuration.extend(["--logdir", logdir])

    tensorboard = tensorboard()
    tensorboard.configure(configuration)
    tensorboard.main()


def flask_main(app, host, port):
    return app.run(host=host, port=port)

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Label studio')
    parser.add_argument(
        '-p', '--port', dest='port', type=int, default=9090,
        help='Server port')
    parser.add_argument(
        '--host', dest='host', type=str, default='0.0.0.0',
        help='Server host')
    parser.add_argument(
        '--kwargs', '--with', dest='kwargs', metavar='KEY=VAL', nargs='+', type=lambda kv: kv.split('='),
        help='Additional AIxBlockMLBase model initialization kwargs')
    parser.add_argument(
        '-d', '--debug', dest='debug', action='store_true',
        help='Switch debug mode')
    parser.add_argument(
        '--log-level', dest='log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default=None,
        help='Logging level')
    parser.add_argument(
        '--model-dir', dest='model_dir', default=os.path.dirname(__file__),
        help='Directory where models are stored (relative to the project directory)')
    parser.add_argument(
        '--check', dest='check', action='store_true',
        help='Validate model instance before launching server')

    args = parser.parse_args()

    # setup logging level
    if args.log_level:
        logging.root.setLevel(args.log_level)

    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def parse_kwargs():
        param = dict()
        for k, v in args.kwargs:
            if v.isdigit():
                param[k] = int(v)
            elif v == 'True' or v == 'true':
                param[k] = True
            elif v == 'False' or v == 'False':
                param[k] = False
            elif isfloat(v):
                param[k] = float(v)
            else:
                param[k] = v
        return param

    kwargs = get_kwargs_from_config()

    if args.kwargs:
        kwargs.update(parse_kwargs())

    if args.check:
        print('Check "' + MyModel.__name__ + '" instance creation..')
        model = MyModel(**kwargs)

    app = init_app(
        model_class=MyModel,
        model_dir=os.environ.get('MODEL_DIR', args.model_dir),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379),
        **kwargs
    )
    from flask import Flask, jsonify, send_from_directory
    # https://stackoverflow.com/questions/55733136/flask-swagger-ui-does-not-recognize-path-to-swagger-json
    from flask_swagger_ui import get_swaggerui_blueprint
    SWAGGER_URL = '/swagger'
    API_URL = '/swagger.json'

    # Thiết lập Swagger UI blueprint
    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={'app_name': "My API"}
    )

    # Đăng ký Swagger UI blueprint
    app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
    # app.run(host=args.host, port=args.port, debug=args.debug, ssl_context=('/app/cert.pem', '/app/privkey.pem'))
    app.run(host=args.host, port=args.port, debug=args.debug)
    CORS(app,resources={r"/*": {"origins": "*"}})

    @app.route('/swagger.json')
    def swagger_spec():
        return send_from_directory(os.getcwd(), 'swagger.json')
else:
    # for uWSGI use
    app = init_app(
        model_class=MyModel,
        model_dir=os.environ.get('MODEL_DIR', os.path.dirname(__file__)),
        redis_queue=os.environ.get('RQ_QUEUE_NAME', 'default'),
        redis_host=os.environ.get('REDIS_HOST', 'localhost'),
        redis_port=os.environ.get('REDIS_PORT', 6379)
    )
    from flask_swagger_ui import get_swaggerui_blueprint
    SWAGGER_URL = '/swagger'
    API_URL = '/swagger.json'

    # Thiết lập Swagger UI blueprint
    swagger_ui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={'app_name': "My API"}
    )

    # Đăng ký Swagger UI blueprint
    app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)
    
    CORS(app,resources={r"/*": {"origins": "*"}})

    @app.route('/swagger.json')
    def swagger_spec():
        return send_from_directory(os.getcwd(), 'swagger.json')
    
    @app.route('/downloads', methods=['GET'])
    def download_file():
        # Lấy tên file từ query parameter
        filename = request.args.get('path')
        if not filename:
            return abort(400, description="File name is required")
        
        # Tạo đường dẫn đầy đủ đến file
        current_dir = os.getcwd()
    
        # Tạo đường dẫn đầy đủ đến file
        full_path = os.path.join(current_dir, filename)
        
        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(full_path):
            return abort(404, description="File not found")
        
        # Trả về file dưới dạng đính kèm
        return send_file(full_path, as_attachment=True)

    @app.route('/documents')
    def serve_index():
        # Trả về file index.html từ thư mục hiện tại
        return send_from_directory(os.getcwd(), 'index.html')
