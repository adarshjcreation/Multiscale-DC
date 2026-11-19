"""Check the field test patches in data/test against their MD5 checksums.  Usage: python scripts/check_data.py

The field data cannot be redistributed and are not part of the repository (see data/README.md).
"""
import hashlib
import os
import sys

MD5 = {
    "data14.mat": "63151da22fbb3f98b02fab58e5fc2933",
    "data16.mat": "0a5947cfbf9c35dff41107a61a6c1099",
    "data18.mat": "52a7baad81ff0aa151b2e0b435bad4f0",
    "data20.mat": "995bb1c4b8e2840ddcbd59487d5ac2ce",
    "data22.mat": "d69eafc6ae3379e7c99fb16536d31d73",
    "data24.mat": "627f7b19adcdedcaed9256dcda69221d",
    "data26.mat": "eb2f9d95a8c72984a817138a1acfd82c",
    "data28.mat": "b1318531e2e7ac557185a6d6a8479c8a",
    "data30.mat": "c9a2fa3e7b13f009d33d4c70566e220a",
    "data32.mat": "83ee18dc7e18428c1504ed89eb53882f",
}
folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")
ok = True
for name, ref in MD5.items():
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"MISSING  {name}")
        ok = False
        continue
    with open(path, "rb") as f:
        good = hashlib.md5(f.read()).hexdigest() == ref
    ok &= good
    print(f"{'OK      ' if good else 'BAD MD5 '} {name}")
extra = sorted(set(f for f in os.listdir(folder) if f.endswith(".mat")) - set(MD5)) if os.path.isdir(folder) else []
if extra:
    print("unexpected files (the notebooks read every .mat file in data/test):", ", ".join(extra))
    ok = False
if not ok:
    print("The field data are not included in the repository; see data/README.md.")
sys.exit(0 if ok else 1)
