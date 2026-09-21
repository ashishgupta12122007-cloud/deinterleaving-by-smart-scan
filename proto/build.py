import json, sys, os
here = os.path.dirname(os.path.abspath(__file__))
data = open(os.path.join(here, "..", "out", "prototype_data.json")).read()
data = data.replace("</", "<\\/").replace("<!--", "<\\!--")
tpl = open(os.path.join(here, "template.html")).read()
assert "/*__DATA__*/" in tpl
out = tpl.replace("/*__DATA__*/", data)
dst = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "sih26055_smart_scan.html")
open(dst, "w").write(out)
print("built", dst, len(out)//1024, "KB")
