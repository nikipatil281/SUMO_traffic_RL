# scripts/make_grid_net.py
import os, subprocess, textwrap, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
NETDIR = os.path.join(ROOT, "envs", "grid2x2")  # keep same folder name
os.makedirs(NETDIR, exist_ok=True)

NET = os.path.join(NETDIR, "net.net.xml")
ROU = os.path.join(NETDIR, "routes.rou.xml")
CFG = os.path.join(NETDIR, "grid.sumocfg")
TMP = os.path.join(NETDIR, "net.tls.net.xml")

GRID_N = 3  # 3x3 grid so there are real intersections

def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)

def write(path, s):
    with open(path, "w") as f:
        f.write(s)
    print("wrote:", path)

def ensure_net():
    # 1) generate a plain grid
    if os.path.exists(NET): os.remove(NET)
    run([
        "netgenerate",
        "--grid", f"--grid.number={GRID_N}",
        "--default.lanenumber=1",
        "--default.speed=13.89",
        "-o", NET
    ])
    # 2) add traffic-light programs by guessing & rebuilding (this creates tlLogic)
    #    lower threshold -> more intersections get a TLS in tiny toy nets
    if os.path.exists(TMP): os.remove(TMP)
    run([
        "netconvert",
        "--sumo-net-file", NET,
        "--tls.guess",
        "--tls.guess.threshold", "1",
        "--tls.default-type", "static",
        "--tls.rebuild",
        "-o", TMP
    ])
    # replace original file with tls-augmented one
    os.replace(TMP, NET)

    # Summarize TLS count
    try:
        import sumolib
        net = sumolib.net.readNet(NET)
        tls_ids = [tl.getID() for tl in net.getTrafficLights()]
        print(f"TLS count: {len(tls_ids)}  -> {tls_ids}")
    except Exception as e:
        print("note: could not inspect TLS via sumolib:", e)

def ensure_routes():
    if os.path.exists(ROU): os.remove(ROU)
    routes_template = textwrap.dedent("""\
    <routes>
      <vType id="car" accel="2.0" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" />
      <flow id="w2e" type="car" begin="0" end="3600" probability="0.05" from="left0" to="right1"/>
      <flow id="e2w" type="car" begin="0" end="3600" probability="0.05" from="right1" to="left0"/>
      <flow id="s2n" type="car" begin="0" end="3600" probability="0.05" from="bottom0" to="top1"/>
      <flow id="n2s" type="car" begin="0" end="3600" probability="0.05" from="top1" to="bottom0"/>
    </routes>
    """)
    try:
        import sumolib
        net = sumolib.net.readNet(NET)
        def border_edges():
            edges = [e for e in net.getEdges() if not e.getID().startswith(":")]
            xs, ys = [], []
            for n in net.getNodes():
                x,y = n.getCoord(); xs.append(x); ys.append(y)
            xmin,xmax = min(xs), max(xs); ymin,ymax = min(ys), max(ys)
            left,right,top,bottom = [],[],[],[]
            for e in edges:
                fx,fy = e.getFromNode().getCoord(); tx,ty = e.getToNode().getCoord()
                if min(fx,tx) == xmin and max(fx,tx) > xmin: left.append(e)
                if max(fx,tx) == xmax and min(fx,tx) < xmax: right.append(e)
                if min(fy,ty) == ymin and max(fy,ty) > ymin: bottom.append(e)
                if max(fy,ty) == ymax and min(fy,ty) < ymax: top.append(e)
            def pick(edge_list):
                for e in edge_list:
                    fx,fy = e.getFromNode().getCoord(); tx,ty = e.getToNode().getCoord()
                # pick any with nonzero delta
                    if (tx,ty) != (fx,fy): return e
                return edge_list[0]
            return pick(left), pick(right), pick(top), pick(bottom)
        left,right,top,bottom = border_edges()
        txt = (routes_template
               .replace("left0", left.getID())
               .replace("right1", right.getID())
               .replace("top1", top.getID())
               .replace("bottom0", bottom.getID()))
        write(ROU, txt)
    except Exception as e:
        print("warning: sumolib route mapping failed, writing template routes:", e)
        write(ROU, routes_template)

def ensure_cfg():
    if os.path.exists(CFG): os.remove(CFG)
    cfg = f"""\
    <configuration>
      <input>
        <net-file value="net.net.xml"/>
        <route-files value="routes.rou.xml"/>
      </input>
      <time>
        <step-length value="1.0"/>
      </time>
    </configuration>
    """
    write(CFG, cfg)

if __name__ == "__main__":
    ensure_net()
    ensure_routes()
    ensure_cfg()
    print("done. Files in:", NETDIR)
