from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler
from pyftpdlib.servers import FTPServer

authorizer = DummyAuthorizer()
authorizer.add_user("vadim", "asusp535", "/home/vadim/hackatones/medhack/src/ftp_server/", perm="elradfmw")
authorizer.add_anonymous("/home/vadim/hackatones/medhack/src/ftp_server/test/", perm="elradfmw")

handler = FTPHandler
handler.authorizer = authorizer

server = FTPServer(("10.20.1.163", 8081), handler)
server.serve_forever()
