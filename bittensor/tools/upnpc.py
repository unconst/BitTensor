import argparse
import miniupnpc
from loguru import logger

parser = argparse.ArgumentParser(description='UPnP Tool.')
parser.add_argument('--port')
args = parser.parse_args()

# A Tool for punching a hole in UPNPC enabled routers.
def main(args):
    try:
        u = miniupnpc.UPnP()
        logger.info('Using UPnP for port mapping...')
        logger.info('Discovering... delay=%ums' % u.discoverdelay)
        u.discoverdelay = 200
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

        logger.info('trying to redirect %s port %u TCP => %s port %u TCP' % (external_ip, external_port, local_ip, local_port))
        b = u.addportmapping(external_port, 'TCP', local_ip, local_port, 'UPnP IGD Tester port %u' % external_port, '')

    except Exception as e:
        logger.error('Exception in UPnP :', e)
        exit(1)

    print ('--external_ip=' + str(external_ip) + ' --external_port=' + str(external_port) +' --local_ip=' + str(local_ip) + ' --local_port=' + str(local_port))


if __name__ == '__main__':
    main(args)
