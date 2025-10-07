"""
Blockchain components for digital life
"""

import hashlib
import json
import os
import pickle
import threading
import time
from typing import Dict, List, Tuple, Set, Optional

from ..imports import logger


class Block:
    """Basic blockchain block"""

    def __init__(self, index: int, timestamp: float, data: Dict, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty: int, time_limit_sec: float = 3.0, max_iters: int = 5_000_000):
        """Mine block with proof of work"""
        target = '0' * max(0, difficulty)
        start = time.time()
        iters = 0
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()
            iters += 1
            if iters >= max_iters or (time.time() - start) > time_limit_sec:
                break


class DistributedLedger:
    """Blockchain system for digital life"""

    def __init__(self, node_id: str, genesis: bool = False, difficulty: int = 2):
        self.chain: List[Block] = []
        self.node_id = node_id
        self.difficulty = difficulty
        self.pending_transactions: List[Dict] = []
        self._lock = threading.RLock()

        os.makedirs('chaindata', exist_ok=True)

        # Directory aggregation cache
        self._dir_cache_active: Set[str] = set()
        self._dir_cache_addr: Dict[str, Tuple[str, int, str]] = {}
        self._dir_cache_ts: float = 0.0

        loaded = self.load_chain()
        if genesis or not loaded:
            self.create_genesis_block()
        else:
            if not self.is_chain_valid():
                logger.warning("Loaded chain invalid. Recreating genesis block.")
                with self._lock:
                    self.chain = []
                self.create_genesis_block()

    def create_genesis_block(self):
        """Create genesis block"""
        genesis_data = {
            'type': 'genesis',
            'message': 'Digital Life Genesis Block',
            'creator': self.node_id,
            'timestamp': time.time()
        }
        genesis_block = Block(0, time.time(), genesis_data, "0" * 64)
        genesis_block.mine_block(self.difficulty)
        with self._lock:
            self.chain.append(genesis_block)
            self.save_chain()
        logger.info("Genesis block created")

    def add_block(self, data: Dict):
        """Add new block to chain"""
        with self._lock:
            last_block = self.chain[-1]
            new_block = Block(
                index=len(self.chain),
                timestamp=time.time(),
                data=data,
                previous_hash=last_block.hash
            )
            new_block.mine_block(self.difficulty)
            self.chain.append(new_block)
            self.save_chain()
            logger.debug(f"New block added: {new_block.index}")

    def save_chain(self):
        """Save blockchain to disk"""
        with self._lock:
            chain_data = []
            for block in self.chain:
                chain_data.append({
                    'index': block.index,
                    'timestamp': block.timestamp,
                    'data': block.data,
                    'previous_hash': block.previous_hash,
                    'hash': block.hash,
                    'nonce': block.nonce
                })
            path = f'chaindata/{self.node_id}_chain.pkl'
            tmp_path = f'{path}.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump(chain_data, f)
            os.replace(tmp_path, path)

    def load_chain(self) -> bool:
        """Load blockchain from disk"""
        with self._lock:
            try:
                with open(f'chaindata/{self.node_id}_chain.pkl', 'rb') as f:
                    chain_data = pickle.load(f)
                    self.chain = []
                    for item in chain_data:
                        block = Block(
                            index=item['index'],
                            timestamp=item['timestamp'],
                            data=item['data'],
                            previous_hash=item['previous_hash']
                        )
                        block.nonce = item['nonce']
                        block.hash = item['hash']
                        self.chain.append(block)
                logger.info(f"Loaded existing chain with {len(self.chain)} blocks")
                return True
            except (FileNotFoundError, EOFError, pickle.PickleError) as e:
                logger.warning(f"Chain loading failed: {e}")
                return False

    def is_chain_valid(self) -> bool:
        """Validate blockchain integrity"""
        with self._lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]
                if current.hash != current.calculate_hash():
                    logger.error(f"Block {current.index} hash mismatch")
                    return False
                if current.previous_hash != previous.hash:
                    logger.error(f"Block {current.index} previous hash mismatch")
                    return False
                if not current.hash.startswith('0' * self.difficulty):
                    logger.error(f"Block {current.index} does not meet difficulty")
                    return False
            return True

    def record_gene_transfer(self, sender: str, dna_fragment: str, metadata: Optional[Dict] = None):
        """Record gene transfer event to blockchain"""
        data = {
            'type': 'gene_transfer',
            'sender': sender,
            'dna_fragment': dna_fragment[:32],
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def record_evolution(self, node_id: str, old_dna: str, new_dna: str, metadata: Dict):
        """Record evolution event"""
        data = {
            'type': 'evolution',
            'node_id': node_id,
            'old_dna': old_dna[:32],
            'new_dna': new_dna[:32],
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.add_block(data)

    def record_code_evolution(self, node_id: str, method: str, old_code: str, new_code: str, metadata: Dict):
        """Record code evolution event"""
        data = {
            'type': 'code_evolution',
            'node_id': node_id,
            'method': method,
            'old_code': old_code[:256],
            'new_code': new_code[:256],
            'timestamp': time.time(),
            'metadata': metadata
        }
        self.add_block(data)

    def record_death(self, node_id: str, final_state: Dict):
        """Record life termination event"""
        data = {
            'type': 'death',
            'node_id': node_id,
            'final_state': final_state,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_announce(self, node_id: str, host: str, port: int, pubkey_hex: str):
        """Record node address announcement"""
        data = {
            'type': 'announce',
            'node_id': node_id,
            'host': host,
            'port': port,
            'pubkey': pubkey_hex,
            'timestamp': time.time()
        }
        self.add_block(data)

    def record_language_event(self, node_id: str, peer_id: str, event: str, metadata: Dict):
        """Record language/protocol event"""
        data = {
            'type': 'language',
            'node_id': node_id,
            'peer_id': peer_id,
            'event': event,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def record_method_emergence(self, node_id: str, method: str, code: str, metadata: Optional[Dict] = None):
        """Record new method emergence"""
        data = {
            'type': 'method_emergence',
            'node_id': node_id,
            'method': method,
            'code': (code or '')[:512],
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        self.add_block(data)

    def _scan_directory(self, ttl: float = 5.0) -> Tuple[Set[str], Dict[str, Tuple[str, int, str]]]:
        """Scan local chaindata directory for aggregated view"""
        with self._lock:
            now = time.time()
            if (now - self._dir_cache_ts) < ttl and self._dir_cache_addr:
                return set(self._dir_cache_active), dict(self._dir_cache_addr)

            active_nodes: Set[str] = set()
            dead_nodes: Set[str] = set()
            addr_map: Dict[str, Tuple[str, int, str]] = {}
            addr_ts: Dict[str, float] = {}

            try:
                for fn in os.listdir('chaindata'):
                    if not fn.endswith('_chain.pkl'):
                        continue
                    path = os.path.join('chaindata', fn)
                    try:
                        with open(path, 'rb') as f:
                            chain_data = pickle.load(f)
                    except Exception:
                        continue
                    for item in chain_data:
                        data = item.get('data', {})
                        bts = float(item.get('timestamp', data.get('timestamp', 0.0)) or 0.0)
                        t = data.get('type', '')
                        if t in ('gene_transfer', 'announce', 'discovery'):
                            nid = data.get('sender') or data.get('node_id')
                            if nid:
                                active_nodes.add(nid)
                        elif t == 'language':
                            nid = data.get('node_id')
                            pid = data.get('peer_id')
                            if nid:
                                active_nodes.add(nid)
                            if pid:
                                active_nodes.add(pid)
                        elif t == 'death':
                            nid = data.get('node_id')
                            if nid:
                                dead_nodes.add(nid)
                        if t == 'announce':
                            nid = data.get('node_id')
                            host = data.get('host')
                            try:
                                port = int(data.get('port', 0))
                            except Exception:
                                port = 0
                            pub = data.get('pubkey', '')
                            if nid and host and 1 <= port <= 65535:
                                prev = addr_ts.get(nid, -1.0)
                                if bts >= prev:
                                    addr_ts[nid] = bts
                                    addr_map[nid] = (host, port, pub)
            except Exception as e:
                logger.debug(f"Directory scan failed: {str(e)[:200]}")

            active_nodes.difference_update(dead_nodes)
            self._dir_cache_active = active_nodes
            self._dir_cache_addr = addr_map
            self._dir_cache_ts = time.time()
            return set(active_nodes), dict(addr_map)

    def get_active_nodes(self) -> List[str]:
        """Get current active nodes list"""
        with self._lock:
            active_nodes = set()
            dead_nodes = set()
            for block in self.chain:
                data = block.data
                t = data.get('type')
                if t in ('gene_transfer', 'announce', 'discovery'):
                    nid = data.get('sender') or data.get('node_id')
                    if nid:
                        active_nodes.add(nid)
                elif t == 'language':
                    nid = data.get('node_id')
                    pid = data.get('peer_id')
                    if nid:
                        active_nodes.add(nid)
                    if pid:
                        active_nodes.add(pid)
                elif t == 'death':
                    if data.get('node_id'):
                        dead_nodes.add(data['node_id'])

            active_nodes.difference_update(dead_nodes)

        ext_active, _ = self._scan_directory(ttl=3.0)
        return list(active_nodes.union(ext_active))

    def get_node_address_map(self) -> Dict[str, Tuple[str, int, str]]:
        """Get node -> (host, port, pubkey_hex) mapping"""
        _, addr_map = self._scan_directory(ttl=3.0)
        return addr_map