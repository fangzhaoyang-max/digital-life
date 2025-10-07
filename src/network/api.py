"""
Network communication API for digital life
"""

import base64
import hashlib
import json
import threading
import time
from typing import Dict, Optional

import requests
from flask import Flask, jsonify, request
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..utils.common import is_allowed_destination, json_sanitize
from ..imports import logger


class NetworkAPI:
    """Network communication API for digital life"""

    def __init__(self, digital_life_instance):
        self.owner = digital_life_instance
        self.app = Flask(__name__)
        try:
            self.app.config['MAX_CONTENT_LENGTH'] = int(self.owner.config.get('max_payload_bytes') or 0)
        except Exception:
            self.app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
        self._init_routes()

    def _require_auth(self, f):
        from functools import wraps

        @wraps(f)
        def wrapper(*args, **kwargs):
            token = request.headers.get('X-Auth-Token')
            if token != self.owner.config['auth_token']:
                return jsonify({'status': 'unauthorized'}), 401
            if self.owner.config.get('allowlist') and request.remote_addr not in self.owner.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            return f(*args, **kwargs)

        return wrapper

    def _init_routes(self):
        """Initialize API routes"""

        @self.app.route('/ping', methods=['GET'])
        def ping():
            return jsonify({
                'status': self.owner.state.name,
                'node': self.owner.node_id,
                'consciousness': self.owner.consciousness_level,
                'energy': self.owner.energy,
                'code_version': self.owner.code_version
            })

        @self.app.route('/exchange_dna', methods=['POST'])
        @self._require_auth
        def exchange_dna():
            data = request.json or {}
            dna = data.get('dna', '')
            if self._validate_dna(dna):
                threading.Thread(target=self._horizontal_gene_transfer, args=(dna, data.get('metadata', {})), daemon=True).start()
                return jsonify({'status': 'accepted'})
            return jsonify({'status': 'invalid_dna'}), 400

        @self.app.route('/replicate', methods=['POST'])
        @self._require_auth
        def replicate():
            if self.owner.energy > self.owner.config['energy_threshold']:
                data = request.json or {}
                threading.Thread(target=self._assimilate, args=(data,), daemon=True).start()
                return jsonify({'status': 'replication_started'})
            return jsonify({'status': 'low_energy'}), 400

        @self.app.route('/learn', methods=['POST'])
        @self._require_auth
        def learn():
            knowledge = (request.json or {}).get('knowledge', {})
            if knowledge:
                threading.Thread(target=self._integrate_knowledge, args=(knowledge,), daemon=True).start()
                return jsonify({'status': 'learning_started'})
            return jsonify({'status': 'no_knowledge'}), 400

        @self.app.route('/get_code', methods=['GET'])
        @self._require_auth
        def get_code():
            method = request.args.get('method')
            if method in getattr(self.owner, 'mutable_methods', []):
                try:
                    code = self.owner._method_sources.get(method) or __import__('inspect').getsource(getattr(self.owner, method))
                except Exception:
                    code = self.owner._method_sources.get(method, '')
                return jsonify({
                    'method': method,
                    'code': code,
                    'version': self.owner.code_engine.code_versions.get(method, 1)
                })
            return jsonify({'status': 'invalid_method'}), 404

        @self.app.route('/receive_code', methods=['POST'])
        @self._require_auth
        def receive_code():
            if self.owner.state.name == 'REPLICATING':
                return jsonify({'status': 'busy_replicating'}), 400
            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400
            threading.Thread(target=self.owner._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

        @self.app.route('/receive_code_signed', methods=['POST'])
        def receive_code_signed():
            if self.owner.config.get('allowlist') and request.remote_addr not in self.owner.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403
            if self.owner.state.name == 'REPLICATING':
                return jsonify({'status': 'busy_replicating'}), 400

            code_data = request.json
            if not code_data or 'payload' not in code_data or 'sig' not in code_data or 'pubkey' not in code_data:
                return jsonify({'status': 'invalid_code'}), 400

            sig_hex = code_data.get('sig', '')
            pubkey_hex = code_data.get('pubkey', '')
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and all(c in '0123456789abcdefABCDEF' for c in sig_hex)):
                return jsonify({'status': 'bad_signature_format'}), 400
            if not (isinstance(pubkey_hex, str) and len(pubkey_hex) == 64 and all(c in '0123456789abcdefABCDEF' for c in pubkey_hex)):
                return jsonify({'status': 'bad_pubkey_format'}), 400

            try:
                payload_b64 = code_data['payload']
                payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
                digest = hashlib.sha256(payload_bytes).digest()
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
                data = json.loads(payload_bytes.decode('utf-8'))
                source_node = data.get('metadata', {}).get('source_node', '')
                addr_map = self.owner.blockchain.get_node_address_map()
                if source_node and source_node not in addr_map:
                    host = data.get('metadata', {}).get('host') or request.remote_addr
                    port = int(data.get('metadata', {}).get('port', 0))
                    if 1 <= port <= 65535:
                        if is_allowed_destination(host, self.owner.config.get('strict_target_ip_check', True), self.owner.config.get('local_discovery_enable', True)):
                            try:
                                self.owner.blockchain.record_announce(source_node, host, port, pubkey_hex)
                                logger.info(f"Auto-registered announce for {source_node} at {host}:{port}")
                            except Exception:
                                pass
            except Exception:
                pass

            threading.Thread(target=self.owner._integrate_code, args=(code_data,), daemon=True).start()
            return jsonify({'status': 'code_received'})

        @self.app.route('/speak_signed', methods=['POST'])
        def speak_signed():
            if self.owner.config.get('allowlist') and request.remote_addr not in self.owner.config['allowlist']:
                return jsonify({'status': 'forbidden'}), 403

            pkt = request.json
            if not pkt or 'payload' not in pkt or 'sig' not in pkt or 'pubkey' not in pkt:
                return jsonify({'status': 'invalid'}), 400
            sig_hex = pkt.get('sig', '')
            pub_hex = pkt.get('pubkey', '')
            if not (isinstance(sig_hex, str) and len(sig_hex) == 128 and all(c in '0123456789abcdefABCDEF' for c in sig_hex)):
                return jsonify({'status': 'bad_signature_format'}), 400
            if not (isinstance(pub_hex, str) and len(pub_hex) == 64 and all(c in '0123456789abcdefABCDEF' for c in pub_hex)):
                return jsonify({'status': 'bad_pubkey_format'}), 400

            try:
                payload_b64 = pkt['payload']
                payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
                digest = hashlib.sha256(payload_bytes).digest()
                vk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
                vk.verify(bytes.fromhex(sig_hex), digest)
                data = json.loads(payload_bytes.decode('utf-8'))
            except Exception as e:
                logger.error(f"Speak verify failed: {e}")
                return jsonify({'status': 'verify_failed'}), 400

            try:
                source_node = data.get('meta', {}).get('source_node') or data.get('metadata', {}).get('source_node', '')
                addr_map = self.owner.blockchain.get_node_address_map()
                if source_node and source_node not in addr_map:
                    host = data.get('meta', {}).get('host') or request.remote_addr
                    port = int(data.get('meta', {}).get('port', 0))
                    if 1 <= port <= 65535:
                        if is_allowed_destination(host, self.owner.config.get('strict_target_ip_check', True), self.owner.config.get('local_discovery_enable', True)):
                            try:
                                self.owner.blockchain.record_announce(source_node, host, port, pub_hex)
                                logger.info(f"Auto-registered announce for {source_node} at {host}:{port}")
                            except Exception:
                                pass
                addr_map = self.owner.blockchain.get_node_address_map()
                if source_node in addr_map:
                    _, _, announced_pubkey = addr_map[source_node]
                    if announced_pubkey and announced_pubkey != pub_hex:
                        return jsonify({'status': 'pubkey_mismatch'}), 403
            except Exception:
                pass

            ok = self.owner._process_language_message(data)
            return jsonify({'status': 'ok' if ok else 'accepted'})

        @self.app.route('/language_stats', methods=['GET'])
        def language_stats():
            st = self.owner.language.stats()
            peers = {}
            for pid, ps in list(self.owner.language.peer_state.items())[:50]:
                peers[pid] = {
                    'agreed_version': ps.get('agreed_version'),
                    'success_rate': self.owner.language.peer_success_rate(pid),
                    'history_len': len(ps.get('history', [])),
                    'last_caps': ps.get('last_caps', [])
                }
            return jsonify({
                'node': self.owner.node_id,
                'language': st,
                'talk_prob': self.owner.config.get('language_talk_prob', 0.15),
                'meta': {
                    'learning_rate': self.owner.config.get('learning_rate', 0.001),
                    'memory_interval': self.owner._interval_overrides.get('memory', 20.0),
                    'code_evolution_prob': self.owner.config.get('code_evolution_prob', 0.15),
                },
                'peers': peers
            })

    def _validate_dna(self, dna: str) -> bool:
        """Basic DNA validation"""
        if not isinstance(dna, str) or len(dna) < 32:
            return False
        try:
            int(dna, 16)
        except Exception:
            return False
        return len(dna) % 32 == 0 or len(dna) >= 160

    def _horizontal_gene_transfer(self, donor_dna: str, metadata: Dict):
        try:
            new_dna = self.owner.genetic_encoder.recombine(self.owner.dna, donor_dna)
            self.owner.dna = new_dna
            logger.info("Horizontal gene transfer completed")
        except Exception as e:
            logger.error(f"Gene transfer failed: {e}")

    def _assimilate(self, data: Dict):
        self.owner._integrate_code(data)

    def _integrate_knowledge(self, knowledge: Dict):
        if isinstance(knowledge, dict):
            with self.owner._kb_lock:
                self.owner.knowledge_base.update(knowledge)
                self.owner._prune_knowledge_base(max_items=2000)
            logger.info("Knowledge integrated")

    def run_server(self):
        """Run the API server"""
        try:
            logger.info(f"API server starting on {self.owner.config['host']}:{self.owner.config['port']}")
            self.app.run(
                host=self.owner.config['host'], 
                port=self.owner.config['port'], 
                debug=False, 
                threaded=True, 
                use_reloader=False
            )
        except OSError as e:
            msg = str(e).lower()
            if 'address already in use' in msg or '10048' in msg or '98' in msg:
                from ..utils.common import find_free_port
                new_port = find_free_port(self.owner.config['port'] + 1, self.owner.config['port'] + 200)
                logger.warning(f"Port {self.owner.config['port']} busy, retry on {new_port}")
                self.owner.config['port'] = new_port
                try:
                    self.owner.blockchain.record_announce(self.owner.node_id, self.owner.config['host'], self.owner.config['port'], self.owner._pubkey_hex)
                except Exception:
                    pass
                self.app.run(host=self.owner.config['host'], port=new_port, debug=False, threaded=True, use_reloader=False)
            else:
                logger.error(f"API server failed: {e}")
        except Exception as e:
            logger.error(f"API server failed: {e}")

    def send_replication_package(self, target_node: str, package: Dict) -> bool:
        """Send replication package to target node"""
        if not self.owner.config.get('network_enable', True):
            return False
        try:
            addr_map = self.owner.blockchain.get_node_address_map()
            if target_node not in addr_map:
                return False
            host, port, _pub = addr_map[target_node]
            base = f"http://{host}:{port}"

            if not is_allowed_destination(host, self.owner.config.get('strict_target_ip_check', True), self.owner.config.get('local_discovery_enable', True)):
                logger.warning(f"Skip disallowed target address: {host}")
                return False

            try:
                response = requests.post(
                    f"{base}/receive_code_signed",
                    json=package,
                    timeout=5
                )
                return response.status_code == 200
            except Exception:
                return False
        except Exception as e:
            logger.debug(f"Failed to send to {target_node}: {str(e)[:100]}")
            return False

    def send_language_message(self, target_node: str, message: Dict) -> bool:
        """Send language message to target node"""
        if not self.owner.config.get('network_enable', True):
            return False
        try:
            addr_map = self.owner.blockchain.get_node_address_map()
            if target_node not in addr_map:
                return False
            host, port, _ = addr_map[target_node]
            
            if not is_allowed_destination(host, self.owner.config.get('strict_target_ip_check', True), self.owner.config.get('local_discovery_enable', True)):
                return False

            url = f"http://{host}:{port}/speak_signed"
            response = requests.post(url, json=message, timeout=3)
            return response.status_code == 200
        except Exception:
            return False