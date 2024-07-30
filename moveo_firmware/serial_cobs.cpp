#include "serial_cobs.h"

uint8_t pck_tx_buffer[MAX_PACKET_DATA_LEN + 6];

uint8_t CRC8(const uint8_t* data, uint8_t len)
{
    uint8_t* p = (uint8_t*)data;
    uint8_t crc = 0x00;
    uint8_t poly = 0x07;
    uint8_t i;

    while(len--)
    {
        crc = crc ^ *p++;

        for(i=0; i<8; ++i)
        {
            if(crc & 0x80)
                crc = (uint8_t)((crc << 1) ^ poly);
            else
                crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}

void cobsEncode(const uint8_t* in, uint8_t* out, const uint8_t len)
{
    uint8_t blockPtr = 0;
    uint8_t blockLen = 1;
    out[len+1] = 0x00;

    for(uint8_t i=0; i<len; ++i)
    {
        if(in[i] == 0x00)
        {
            out[blockPtr] = blockLen;
            blockPtr += blockLen;
            blockLen = 1;
        }
        else
        {
            out[i+1] = in[i];
            ++blockLen;
        }
    }
    out[blockPtr] = blockLen;
}

bool cobsDecode(const uint8_t* in, uint8_t* out, const uint8_t len)
{
    int8_t blockPtr = 0;
    uint8_t blockLen;
    uint8_t i;
    do
    {
        blockLen = in[blockPtr];
        if(blockPtr + blockLen >= len)
            return false;
        for(i=1; i<blockLen; ++i)
            *out++ = in[blockPtr+i];
        *out++ = 0x00;
        blockPtr += blockLen;
    } while (blockPtr < len-1);

    return true;
}

void encodePacket(const SerialPacket* pck, uint8_t* bytes, uint8_t* len)
{
    //Convert packet to bytes
    uint8_t buffer[4+pck->dataLen];
	buffer[0] = '#';
	buffer[1] = (uint8_t)pck->cmd;
	buffer[2] = pck->dataLen;
	for(uint8_t i=0; i<pck->dataLen; ++i)
		buffer[3+i] = pck->data[i];

	//Append CRC
	buffer[3+pck->dataLen] = CRC8(buffer, 3+pck->dataLen);

	//COBS
	cobsEncode(buffer, bytes, 4+pck->dataLen);
	*len = 4+pck->dataLen+2;
}

bool decodePacket(const uint8_t* bytes, uint8_t len, SerialPacket* pck)
{
    //Check data length
	if(len == 0 || len > 4+MAX_PACKET_DATA_LEN+2)
		return false;

	//COBS
    uint8_t buffer[len];
	if(!cobsDecode(bytes, buffer, len))
        return false;

	//Check preambule
	if(buffer[0] != '$')
		return false;

	//Check CRC
	if(CRC8(buffer, len-3) != buffer[len-3])
		return false;

	//Fill packet
	pck->cmd = byte2Command(buffer[1]);
	pck->dataLen = buffer[2];
	if(pck->dataLen != len-6)
		return false;
	for(uint8_t i=0; i<pck->dataLen; ++i)
		pck->data[i] = buffer[3+i];
	return true;
}

void writePacket(const SerialPacket* pck)
{
    uint8_t len;
    encodePacket(pck, pck_tx_buffer, &len);
    Serial.write(pck_tx_buffer, len);
}