# A Tool for punching a hole in UPNPC enabled routers.

import argparse
import miniupnpc
from loguru import logger

def delete_port_map(args):
    try:
        logger.info('Using UPnP for port mapping...')
        u = miniupnpc.UPnP()
        u.discoverdelay = 200
        logger.info('Discovering... delay=%ums' % u.discoverdelay)
        ndevices = u.discover()
        logger.info(str(ndevices) + ' device(s) detected')
        u.selectigd()
        logger.info('Deleting mapped port=%u' % args.port)
        b = u.deleteportmapping(args.port, 'TCP')
    except Exception as e:
        logger.error('Exception in UPnP :', e)
        exit(1)

def create_port_map(args):
    try:
        u = miniupnpc.UPnP()
        u.discoverdelay = 200
        logger.info('Using UPnP for port mapping...')
        logger.info('Discovering... delay=%ums' % u.discoverdelay)
        ndevices = u.discover()
        logger.info(str(ndevices) + ' device(s) detected')

        u.selectigd()
        local_ip = u.lanaddr
        external_ip = u.externalipaddress()
        local_port = int(args.port)
        external_port = local_port

        logger.info('local ip address :' + str(local_ip))
        logger.info('external ip address :' + str(external_ip))
        logger.info(str(u.statusinfo()) + " " + str(u.connectiontype()))

        # find a free port for the redirection
        rc = u.getspecificportmapping(external_port, 'TCP')
        while rc != None and external_port < 65536:
            external_port += 1
            rc = u.getspecificportmapping(external_port, 'TCP')
        if rc != None:
            logger.error('Exception in UPnP : ' + str(rc))

        logger.info('trying to redirect %s port %u TCP => %s port %u TCP' %
                    (external_ip, external_port, local_ip, local_port))
        b = u.addportmapping(external_port, 'TCP', local_ip, local_port,
                             'UPnP IGD Tester port %u' % external_port, '')

    except Exception as e:
        logger.error('Exception in UPnP :', e)
        exit(1)

    print ('success' + ':' + str(external_ip) + ':' + str(external_port))

def main(args):
    if args.delete == True:
        delete_port_map(args)
    else:
        create_port_map(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPnP Tool.')
    parser.add_argument(
        '--port',
        default=9090,
        type=int,
        help='The port to try porting or deleting Default port=9090')
    parser.add_argument(
        '--delete',
        default=False,
        type=bool,
        help='Delete port or create port. Default delete=False')
    args = parser.parse_args()
    main(args)
