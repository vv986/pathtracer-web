"""开发服务器:禁用缓存,保证改动立即生效。用法: python serve.py [端口]"""
import http.server
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # 安静模式


socketserver.TCPServer.allow_reuse_address = True


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


with Server(('', PORT), Handler) as httpd:
    print(f'serving at http://localhost:{PORT}/ (cache disabled, threaded)')
    httpd.serve_forever()
