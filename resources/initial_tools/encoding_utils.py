"""Encoding, hashing, and cryptographic utilities tool."""

import base64
import hashlib
import hmac
import uuid
import secrets
import string
from typing import Union, Optional


def encoding_utils(
    operation: str,
    data: Union[str, bytes],
    **kwargs
) -> Union[str, bytes, bool]:
    """Perform encoding, hashing, and cryptographic operations.
    
    Args:
        operation: Type of operation ("encode", "decode", "hash", "hmac", "uuid", 
                  "random", "verify_hash")
        data: Input data to process
        **kwargs: Operation-specific parameters
        
    Returns:
        Processed data
        
    Raises:
        ValueError: For invalid operations or parameters
    """
    if operation == "encode":
        encoding = kwargs.get('encoding', 'base64')
        
        # Convert to bytes if string
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        if encoding == 'base64':
            return base64.b64encode(data_bytes).decode('ascii')
        elif encoding == 'base32':
            return base64.b32encode(data_bytes).decode('ascii')
        elif encoding == 'base16' or encoding == 'hex':
            return base64.b16encode(data_bytes).decode('ascii')
        elif encoding == 'url':
            return base64.urlsafe_b64encode(data_bytes).decode('ascii')
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")
    
    elif operation == "decode":
        encoding = kwargs.get('encoding', 'base64')
        
        if not isinstance(data, str):
            raise ValueError("Decode operation requires string input")
        
        try:
            if encoding == 'base64':
                decoded = base64.b64decode(data)
            elif encoding == 'base32':
                decoded = base64.b32decode(data)
            elif encoding == 'base16' or encoding == 'hex':
                decoded = base64.b16decode(data)
            elif encoding == 'url':
                decoded = base64.urlsafe_b64decode(data)
            else:
                raise ValueError(f"Unsupported encoding: {encoding}")
            
            # Try to decode as UTF-8 string, otherwise return bytes
            try:
                return decoded.decode('utf-8')
            except UnicodeDecodeError:
                return decoded
                
        except Exception as e:
            raise ValueError(f"Failed to decode: {str(e)}")
    
    elif operation == "hash":
        algorithm = kwargs.get('algorithm', 'sha256')
        
        # Convert to bytes if string
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
        
        if algorithm == 'md5':
            return hashlib.md5(data_bytes).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(data_bytes).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(data_bytes).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data_bytes).hexdigest()
        elif algorithm == 'blake2b':
            return hashlib.blake2b(data_bytes).hexdigest()
        elif algorithm == 'blake2s':
            return hashlib.blake2s(data_bytes).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    elif operation == "hmac":
        key = kwargs.get('key')
        if not key:
            raise ValueError("HMAC operation requires 'key' parameter")
        
        algorithm = kwargs.get('algorithm', 'sha256')
        
        # Convert to bytes
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data
            
        if isinstance(key, str):
            key_bytes = key.encode('utf-8')
        else:
            key_bytes = key
        
        if algorithm == 'sha256':
            return hmac.new(key_bytes, data_bytes, hashlib.sha256).hexdigest()
        elif algorithm == 'sha1':
            return hmac.new(key_bytes, data_bytes, hashlib.sha1).hexdigest()
        elif algorithm == 'sha512':
            return hmac.new(key_bytes, data_bytes, hashlib.sha512).hexdigest()
        elif algorithm == 'md5':
            return hmac.new(key_bytes, data_bytes, hashlib.md5).hexdigest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    elif operation == "uuid":
        version = kwargs.get('version', 4)
        
        if version == 1:
            return str(uuid.uuid1())
        elif version == 4:
            return str(uuid.uuid4())
        elif version == 'name':
            namespace = kwargs.get('namespace', uuid.NAMESPACE_DNS)
            name = kwargs.get('name')
            if not name:
                raise ValueError("UUID name version requires 'name' parameter")
            return str(uuid.uuid5(namespace, name))
        else:
            raise ValueError(f"Unsupported UUID version: {version}")
    
    elif operation == "random":
        type_param = kwargs.get('type', 'string')
        length = kwargs.get('length', 16)
        
        if type_param == 'string':
            charset = kwargs.get('charset', 'alphanumeric')
            if charset == 'alphanumeric':
                chars = string.ascii_letters + string.digits
            elif charset == 'letters':
                chars = string.ascii_letters
            elif charset == 'digits':
                chars = string.digits
            elif charset == 'hex':
                chars = string.hexdigits.lower()
            elif charset == 'ascii':
                chars = string.ascii_letters + string.digits + string.punctuation
            else:
                chars = charset  # Use custom charset
            
            return ''.join(secrets.choice(chars) for _ in range(length))
        
        elif type_param == 'bytes':
            return secrets.token_bytes(length)
        
        elif type_param == 'hex':
            return secrets.token_hex(length)
        
        elif type_param == 'url':
            return secrets.token_urlsafe(length)
        
        elif type_param == 'number':
            max_val = kwargs.get('max', 1000000)
            return secrets.randbelow(max_val)
        
        else:
            raise ValueError(f"Unsupported random type: {type_param}")
    
    elif operation == "verify_hash":
        hash_value = kwargs.get('hash')
        algorithm = kwargs.get('algorithm', 'sha256')
        
        if not hash_value:
            raise ValueError("Verify_hash operation requires 'hash' parameter")
        
        computed_hash = encoding_utils("hash", data, algorithm=algorithm)
        return computed_hash.lower() == hash_value.lower()
    
    elif operation == "verify_hmac":
        signature = kwargs.get('signature')
        key = kwargs.get('key')
        algorithm = kwargs.get('algorithm', 'sha256')
        
        if not signature:
            raise ValueError("Verify_hmac operation requires 'signature' parameter")
        if not key:
            raise ValueError("Verify_hmac operation requires 'key' parameter")
        
        computed_hmac = encoding_utils("hmac", data, key=key, algorithm=algorithm)
        return hmac.compare_digest(computed_hmac.lower(), signature.lower())
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")