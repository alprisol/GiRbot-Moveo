#line 1 "/home/cirs/moveo/moveo_firmware/serial_cobs.h"
#ifndef __SERIAL_COBS_H__
#define __SERIAL_COBS_H__

#include "serial_defs.h"

#define MAX_PACKET_DATA_LEN     64 

//Packet structure: | PREAMBULE = $/# | CMD | DATA_LEN | DATA[0] .... DATA[DATA_LEN-1] | CRC8 |
typedef struct
{
    CommandCode cmd;
    uint8_t dataLen;
    uint8_t data[MAX_PACKET_DATA_LEN];
    
} SerialPacket;

uint8_t CRC8(const uint8_t* data, uint8_t len);
void cobsEncode(const uint8_t* in, uint8_t* out, const uint8_t len);
bool cobsDecode(const uint8_t* in, uint8_t* out, const uint8_t len);

void encodePacket(const SerialPacket* pck, uint8_t* bytes, uint8_t* len);
bool decodePacket(const uint8_t* bytes, uint8_t len, SerialPacket* pck);

void writePacket(const SerialPacket* pck);

#endif