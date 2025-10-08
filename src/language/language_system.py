"""
Language system for digital life communication
"""

import hashlib
import json
import random
import time
from collections import deque
from typing import Dict, List, Set, Optional, Tuple, Any

from ..utils.common import _clamp
from ..imports import logger


class ProtocolRegistry:
    """Manage language protocol versions and syntax"""

    def __init__(self):
        self.specs: Dict[str, Dict] = {}
        self.schema_ids: Dict[str, str] = {}
        self._init_default_specs()

    def _canonical(self, spec: Dict) -> str:
        return json.dumps(spec, sort_keys=True, separators=(',', ':'))

    def register(self, version: str, spec: Dict):
        self.specs[version] = spec
        self.schema_ids[version] = 'proto:' + version + ':' + hashlib.sha1(self._canonical(spec).encode()).hexdigest()[:8]

    def get_schema_id(self, version: str) -> Optional[str]:
        return self.schema_ids.get(version)

    def get_spec(self, version: str) -> Optional[Dict]:
        return self.specs.get(version)

    def validate(self, version: str, message: Dict) -> Tuple[bool, Optional[str]]:
        """Validate message against protocol spec"""
        spec = self.specs.get(version)
        if not spec:
            return True, None
        if version == 'v1':
            return True, None  # v1 is token-only, no structure validation
        intent = message.get('intent')
        slots = message.get('slots', {}) or {}
        intents = spec.get('intents', {})
        if intent not in intents:
            return False, 'unknown_intent'
        want = intents[intent].get('slots', {})
        for k, info in want.items():
            if info.get('required') and k not in slots:
                return False, f'missing_slot:{k}'
        for k, v in slots.items():
            dtype = want.get(k, {}).get('type', 'any')
            if dtype == 'str' and not isinstance(v, str):
                return False, f'bad_type:{k}'
            if dtype == 'int' and not isinstance(v, int):
                return False, f'bad_type:{k}'
            if dtype == 'float' and not isinstance(v, (int, float)):
                return False, f'bad_type:{k}'
        return True, None

    def _init_default_specs(self):
        self.register('v1', {
            'version': 1,
            'style': 'tokens',
            'intents': {}
        })
        self.register('v2', {
            'version': 2,
            'style': 'slots',
            'intents': {
                'greet': {'slots': {}},
                'ask_status': {'slots': {}},
                'share_knowledge': {
                    'slots': {
                        'key': {'type': 'str', 'required': True},
                        'value': {'type': 'str', 'required': False}
                    }
                },
                'propose_trade': {
                    'slots': {
                        'item': {'type': 'str', 'required': True},
                        'price': {'type': 'float', 'required': True}
                    }
                },
                'farewell': {'slots': {}},
                'negotiate_protocol': {
                    'slots': {
                        'version': {'type': 'str', 'required': True}
                    }
                },
                'negotiate_ack': {
                    'slots': {
                        'version': {'type': 'str', 'required': True},
                        'accepted': {'type': 'str', 'required': True}
                    }
                }
            }
        })


class LanguageSystem:
    """Language system for digital life communication"""
    
    BASE_INTENTS = ['greet', 'ask_status', 'share_knowledge', 'propose_trade', 'farewell']
    PROTO_INTENTS = ['negotiate_protocol', 'negotiate_ack']

    def __init__(self, owner):
        self.owner = owner
        self.language_id = hashlib.sha256(owner.node_id.encode()).hexdigest()[:8]
        self.protocol_version = 1
        self.registry = ProtocolRegistry()
        self.supported_versions = ['v1', 'v2']
        self.utterance_map: Dict[str, Set[str]] = {k: set() for k in (self.BASE_INTENTS + self.PROTO_INTENTS)}
        self.lexicon: Dict[str, str] = {}
        self.culture = {
            'tag': f"C{hashlib.sha1((owner.node_id+'-culture').encode()).hexdigest()[:6]}",
            'memes': {},
            'prestige': random.uniform(0.4, 0.8)
        }
        self.seq = 0
        self.successes: deque[int] = deque(maxlen=300)
        self.conversations = 0
        self.peer_state: Dict[str, Dict[str, Any]] = {}
        self._seed_basic_words()

    def _seed_basic_words(self):
        seeds = {
            'greet': ['hai', 'sal'],
            'ask_status': ['stat?'],
            'share_knowledge': ['know!'],
            'propose_trade': ['swap?'],
            'farewell': ['bye'],
            'negotiate_protocol': ['proto?'],
            'negotiate_ack': ['proto!']
        }
        for intent, toks in seeds.items():
            for t in toks:
                self.utterance_map[intent].add(t)
                self.lexicon[t] = intent
                self.culture['memes'][t] = self.culture['memes'].get(t, 0.5)

    def _new_token(self, intent: str) -> str:
        base = f"{intent}:{time.time_ns()}:{random.getrandbits(32)}"
        tok = hashlib.sha1(base.encode()).hexdigest()[:4]
        return tok

    def _get_peer(self, peer_id: Optional[str]) -> Dict[str, Any]:
        if not peer_id:
            return {}
        st = self.peer_state.get(peer_id)
        if not st:
            st = {
                'history': deque(maxlen=60),
                'agreed_version': None,
                'last_caps': [],
                'last_seen': 0.0
            }
            self.peer_state[peer_id] = st
        return st

    def _choose_version_for_peer(self, peer_id: Optional[str]) -> str:
        st = self._get_peer(peer_id)
        if st.get('agreed_version') in self.supported_versions:
            return st['agreed_version']
        return 'v2' if self.protocol_version >= 2 else 'v1'

    def _advertise_caps(self) -> List[str]:
        return list(self.supported_versions)

    def utter(self, intent: str, topic: Optional[str] = None, content: Optional[Dict] = None, peer_id: Optional[str] = None) -> Dict:
        if intent not in (self.BASE_INTENTS + self.PROTO_INTENTS):
            intent = random.choice(self.BASE_INTENTS)
        if not self.utterance_map[intent]:
            t = self._new_token(intent)
            self.utterance_map[intent].add(t)
            self.lexicon[t] = intent
            self.culture['memes'][t] = 0.4

        tok = max(self.utterance_map[intent], key=lambda x: self.culture['memes'].get(x, 0.1))
        self.seq += 1
        version = self._choose_version_for_peer(peer_id)
        proto_ver_num = 2 if version == 'v2' else 1

        msg_core = {
            'intent': intent,
            'utterance': [tok],
            'topic': topic or '',
            'confidence': 0.8
        }

        if version == 'v2':
            schema_id = self.registry.get_schema_id('v2')
            msg_core['schema_id'] = schema_id
            slots = {}
            if intent == 'share_knowledge':
                if content:
                    try:
                        k, v = next(iter(content.items()))
                        slots['key'] = str(k)
                        slots['value'] = json.dumps(self.owner._json_sanitize(v, max_depth=2), ensure_ascii=False)[:256]
                    except Exception:
                        pass
            elif intent == 'propose_trade':
                slots['item'] = str((content or {}).get('item', 'artifact'))
                slots['price'] = float((content or {}).get('price', random.uniform(1, 10)))
            elif intent in ('negotiate_protocol', 'negotiate_ack'):
                slots.update((content or {}))
                for k in list(slots.keys()):
                    slots[k] = str(slots[k])
            msg_core['slots'] = slots

        msg = {
            'meta': {
                'source_node': self.owner.node_id,
                'language_id': self.language_id,
                'protocol_version': proto_ver_num,
                'proto_version': version,
                'proto_caps': self._advertise_caps(),
                'culture_tag': self.culture['tag'],
                'code_version': self.owner.code_version,
                'timestamp': time.time(),
                'seq': self.seq,
                'host': self.owner.config.get('host'),
                'port': self.owner.config.get('port'),
            },
            'message': msg_core
        }

        if content and intent != 'propose_trade':
            msg['message']['content'] = self.owner._json_sanitize(content, max_depth=3)
        return msg

    def interpret(self, payload: Dict, sender_prestige: float = 0.6) -> Tuple[bool, Dict]:
        try:
            meta = payload.get('meta', {})
            msg = payload.get('message', {})
            utter = msg.get('utterance', [])
            intent_hint = msg.get('intent', None)
            schema_id = msg.get('schema_id', None)
            slots = msg.get('slots', {}) or {}
            source = meta.get('source_node', '')
            peer = self._get_peer(source)
            peer['last_caps'] = list(meta.get('proto_caps', []))
            peer['last_seen'] = time.time()

            used_version = None
            if schema_id == self.registry.get_schema_id('v2'):
                used_version = 'v2'
            else:
                used_version = meta.get('proto_version', 'v1')
                if isinstance(used_version, int):
                    used_version = f'v{used_version}'
                if used_version not in self.supported_versions:
                    used_version = 'v1'

            if intent_hint in ('negotiate_protocol', 'negotiate_ack'):
                if intent_hint == 'negotiate_protocol':
                    proposal = (slots.get('version') if slots else None) or 'v2'
                    accept = proposal in self.supported_versions and self.owner.config.get('language_enable_protocol_upgrade', True)
                    if accept:
                        peer['agreed_version'] = proposal
                        self.owner.blockchain.record_language_event(self.owner.node_id, source, 'protocol_set', {'version': proposal})
                    reply = self.utter(
                        intent='negotiate_ack',
                        topic='protocol',
                        content={'version': proposal, 'accepted': 'true' if accept else 'false'},
                        peer_id=source
                    )
                    return True, {
                        'decoded_intent': intent_hint,
                        'topic': msg.get('topic', ''),
                        'content': {'proposal': proposal, 'accepted': accept},
                        'decided_version': peer.get('agreed_version'),
                        'reply': reply
                    }
                else:
                    ver = (slots.get('version') if slots else None) or 'v2'
                    accepted = str(slots.get('accepted', 'false')).lower() == 'true'
                    if accepted and ver in self.supported_versions:
                        peer['agreed_version'] = ver
                        self.owner.blockchain.record_language_event(self.owner.node_id, source, 'protocol_set', {'version': ver})
                    return True, {
                        'decoded_intent': intent_hint,
                        'topic': msg.get('topic', ''),
                        'content': {'version': ver, 'accepted': accepted},
                        'decided_version': peer.get('agreed_version')
                    }

            success = False
            decoded_intent = None

            if used_version == 'v2':
                ok, err = self.registry.validate('v2', msg)
                if ok:
                    decoded_intent = intent_hint
                    success = True

            if not success:
                if utter:
                    token = str(utter[0])[:16]
                    if token in self.lexicon:
                        decoded_intent = self.lexicon[token]
                        success = True
                        self.culture['memes'][token] = _clamp(self.culture['memes'].get(token, 0.3) + 0.05, 0.0, 1.5)
                    else:
                        target_intent = intent_hint if intent_hint in (self.BASE_INTENTS + self.PROTO_INTENTS) else random.choice(self.BASE_INTENTS)
                        self.lexicon[token] = target_intent
                        self.utterance_map[target_intent].add(token)
                        adopt = random.random() < _clamp(sender_prestige, 0.2, 0.95)
                        self.culture['memes'][token] = 0.3 + (0.3 if adopt else 0.0)
                        decoded_intent = target_intent

            if random.random() < self.owner.config.get('language_culture_drift_prob', 0.01):
                if decoded_intent:
                    t2 = self._new_token(decoded_intent)
                    self.lexicon[t2] = decoded_intent
                    self.utterance_map[decoded_intent].add(t2)
                    self.culture['memes'][t2] = 0.2

            self.conversations += 1
            self.successes.append(1 if success else 0)
            if 'history' in peer:
                peer['history'].append(1 if success else 0)

            return success, {
                'decoded_intent': decoded_intent,
                'topic': msg.get('topic', ''),
                'content': msg.get('content', None),
                'decided_version': peer.get('agreed_version', None),
                'used_version': used_version
            }
        except Exception as e:
            logger.error(f"Language interpret error: {e}")
            return False, {}

    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return float(sum(self.successes)) / len(self.successes)

    def peer_success_rate(self, peer_id: str) -> float:
        st = self._get_peer(peer_id)
        h = st.get('history', [])
        if not h:
            return 0.0
        return float(sum(h)) / len(h)

    def stats(self) -> Dict:
        return {
            'language_id': self.language_id,
            'protocol_version': self.protocol_version,
            'lexicon_size': len(self.lexicon),
            'success_rate_300': self.success_rate(),
            'conversations': self.conversations,
            'supported_versions': self.supported_versions
        }