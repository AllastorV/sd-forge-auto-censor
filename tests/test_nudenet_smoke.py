import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from PIL import Image
import nudenet_detect as nd

boxes = nd.detect(Image.new("RGB", (512, 768), (127, 110, 100)))
print("providers:", nd._get_session().get_providers())
print("boxes:", len(boxes))
for b in boxes[:5]:
    assert 0.0 <= b["x1"] <= 1.0 and 0.0 <= b["score"] <= 1.0
print("NUDENET SMOKE OK")
