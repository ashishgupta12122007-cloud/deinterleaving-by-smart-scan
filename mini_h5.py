"""
Minimal read-only HDF5 reader (pure python + numpy).
Supports: superblock v0/v1, v1 object headers, symbol-table groups (B-tree v1 / SNOD / local heap),
contiguous / compact / chunked layouts, deflate + shuffle filters, fixed-point, float, string,
vlen-string (via global heap), compound and array datatypes, attributes.
Only what is needed to read the SIH config_*.h5 files without h5py.
"""
import struct, zlib
import numpy as np


class H5:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.b = f.read()
        b = self.b
        assert b[:8] == b"\x89HDF\r\n\x1a\n", "not HDF5"
        self.sb_ver = b[8]
        assert self.sb_ver in (0, 1), f"superblock v{self.sb_ver} not supported"
        self.so = b[13]
        self.sl = b[14]
        assert self.so == 8 and self.sl == 8
        off = 24 if self.sb_ver == 0 else 28
        self.base = struct.unpack_from("<Q", b, off)[0]
        off += 32
        # root symbol table entry
        _name, self.root_oh, cache, _res = struct.unpack_from("<QQII", b, off)
        self.root = self._read_object(self.root_oh)
        self._gheap = {}

    # ------------------------------------------------------------------ object headers
    def _read_object(self, addr):
        b = self.b
        ver, _r, nmsg, _rc, hsize = struct.unpack_from("<BBHII", b, addr)
        assert ver == 1, f"object header v{ver} unsupported"
        msgs = []
        blocks = [(addr + 16, hsize)]
        while blocks:
            start, size = blocks.pop(0)
            p = start
            end = start + size
            while p + 8 <= end:
                mtype, msize, mflags = struct.unpack_from("<HHB", b, p)
                data = b[p + 8: p + 8 + msize]
                p += 8 + msize
                if mtype == 0x10:  # continuation
                    caddr, clen = struct.unpack_from("<QQ", data, 0)
                    blocks.append((caddr, clen))
                elif mtype != 0:
                    msgs.append((mtype, data))
        return Obj(self, addr, msgs)

    # ------------------------------------------------------------------ groups
    def _heap_data_addr(self, heap_addr):
        b = self.b
        assert b[heap_addr:heap_addr + 4] == b"HEAP"
        _sz, _fl, daddr = struct.unpack_from("<QQQ", b, heap_addr + 8)
        return daddr

    def _cstr(self, addr):
        e = self.b.index(b"\x00", addr)
        return self.b[addr:e].decode("utf-8", "replace")

    def _walk_group_btree(self, baddr, heap_data, out):
        b = self.b
        assert b[baddr:baddr + 4] == b"TREE", "bad btree"
        ntype, level, nused = struct.unpack_from("<BBH", b, baddr + 4)
        p = baddr + 24
        # keys and children interleaved: key0 child0 key1 child1 ... keyN
        for i in range(nused):
            p += 8  # key
            child = struct.unpack_from("<Q", b, p)[0]
            p += 8
            if level > 0:
                self._walk_group_btree(child, heap_data, out)
            else:
                self._read_snod(child, heap_data, out)

    def _read_snod(self, addr, heap_data, out):
        b = self.b
        assert b[addr:addr + 4] == b"SNOD"
        nsym = struct.unpack_from("<H", b, addr + 6)[0]
        p = addr + 8
        for _ in range(nsym):
            noff, oh, cache, _r = struct.unpack_from("<QQII", b, p)
            name = self._cstr(heap_data + noff)
            out[name] = oh
            p += 40

    def group_children(self, obj):
        st = obj.msg(0x11)
        baddr, haddr = struct.unpack_from("<QQ", st, 0)
        hd = self._heap_data_addr(haddr)
        out = {}
        self._walk_group_btree(baddr, hd, out)
        return out

    # ------------------------------------------------------------------ global heap
    def gheap_obj(self, coll_addr, idx):
        if coll_addr not in self._gheap:
            b = self.b
            assert b[coll_addr:coll_addr + 4] == b"GCOL"
            csize = struct.unpack_from("<Q", b, coll_addr + 8)[0]
            p = coll_addr + 16
            end = coll_addr + csize
            objs = {}
            while p + 16 <= end:
                oidx, refc, _r, osize = struct.unpack_from("<HHIQ", b, p)
                if oidx == 0:
                    break
                objs[oidx] = b[p + 16: p + 16 + osize]
                p += 16 + ((osize + 7) // 8) * 8
            self._gheap[coll_addr] = objs
        return self._gheap[coll_addr][idx]

    # ------------------------------------------------------------------ public helpers
    def __getitem__(self, path):
        obj = self.root
        for part in [p for p in path.split("/") if p]:
            kids = self.group_children(obj)
            obj = self._read_object(kids[part])
        return obj

    def tree(self, obj=None, prefix=""):
        obj = obj or self.root
        res = []
        kids = self.group_children(obj) if obj.msg(0x11) is not None else {}
        for name, oh in kids.items():
            o = self._read_object(oh)
            if o.msg(0x11) is not None:
                res.append((prefix + name + "/", o))
                res += self.tree(o, prefix + name + "/")
            else:
                res.append((prefix + name, o))
        return res


# ---------------------------------------------------------------------- datatypes
def parse_dtype(h5, data, off=0):
    """returns (descr, size, next_offset). descr is a dict."""
    cv = data[off]
    cls = cv & 0x0F
    ver = cv >> 4
    bf = data[off + 1] | (data[off + 2] << 8) | (data[off + 3] << 16)
    size = struct.unpack_from("<I", data, off + 4)[0]
    p = off + 8
    d = {"class": cls, "size": size, "bf": bf}
    if cls == 0:  # fixed point
        d["be"] = bool(bf & 1)
        d["signed"] = bool(bf & 8)
        p += 4
    elif cls == 1:  # float
        d["be"] = bool(bf & 1)
        p += 12
    elif cls == 2:  # time
        p += 4
    elif cls == 3:  # string
        d["pad"] = bf & 0x0F
        d["cset"] = (bf >> 4) & 0x0F
    elif cls == 4:  # bitfield
        p += 4
    elif cls == 5:  # opaque
        tl = bf & 0xFF
        p += tl
    elif cls == 6:  # compound
        nmem = bf & 0xFFFF
        members = []
        for _ in range(nmem):
            e = data.index(b"\x00", p)
            name = data[p:e].decode()
            if ver < 3:
                p = p + ((e - p + 1 + 7) // 8) * 8
                moff = struct.unpack_from("<I", data, p)[0]
                p += 4
                if ver == 1:
                    dimn = data[p]
                    p += 4 + 4 + 4 + 16  # dim, reserved, perm, reserved, dim sizes
            else:
                p = e + 1
                nb = max(1, (size.bit_length() + 7) // 8)
                moff = int.from_bytes(data[p:p + nb], "little")
                p += nb
            sub, p = parse_dtype(h5, data, p)
            members.append((name, moff, sub))
        d["members"] = members
    elif cls == 7:  # reference
        pass
    elif cls == 8:  # enum
        base, p = parse_dtype(h5, data, p)
        nmem = bf & 0xFFFF
        names = []
        for _ in range(nmem):
            e = data.index(b"\x00", p)
            names.append(data[p:e].decode())
            p = p + ((e - p + 1 + 7) // 8) * 8 if ver < 3 else e + 1
        vals = []
        for _ in range(nmem):
            vals.append(int.from_bytes(data[p:p + base["size"]], "little"))
            p += base["size"]
        d["base"] = base
        d["enum"] = dict(zip(vals, names))
    elif cls == 9:  # vlen
        d["vtype"] = bf & 0x0F  # 0 sequence, 1 string
        base, p = parse_dtype(h5, data, p)
        d["base"] = base
    elif cls == 10:  # array
        if ver == 2:
            nd = data[p]
            p += 4
        else:
            nd = data[p]
            p += 1
        dims = struct.unpack_from("<" + "I" * nd, data, p)
        p += 4 * nd
        if ver == 2:
            p += 4 * nd
        base, p = parse_dtype(h5, data, p)
        d["dims"] = dims
        d["base"] = base
    return d, p


def np_dtype(d):
    c = d["class"]
    if c == 0:
        return np.dtype(("%s%s%d" % (">" if d["be"] else "<", "i" if d["signed"] else "u", d["size"])))
    if c == 1:
        return np.dtype("%sf%d" % (">" if d["be"] else "<", d["size"]))
    if c == 3:
        return np.dtype("S%d" % d["size"])
    if c == 4:
        return np.dtype("u%d" % d["size"])
    if c == 8:
        return np_dtype(d["base"])
    if c == 6:
        return np.dtype({"names": [m[0] for m in d["members"]],
                         "formats": [np_dtype(m[2]) for m in d["members"]],
                         "offsets": [m[1] for m in d["members"]],
                         "itemsize": d["size"]})
    if c == 10:
        return np.dtype((np_dtype(d["base"]), d["dims"]))
    if c == 9:
        return np.dtype("V16")
    raise NotImplementedError(f"dtype class {c}")


# ---------------------------------------------------------------------- object wrapper
class Obj:
    def __init__(self, h5, addr, msgs):
        self.h5, self.addr, self.msgs = h5, addr, msgs

    def msg(self, t):
        for mt, d in self.msgs:
            if mt == t:
                return d
        return None

    def msgs_of(self, t):
        return [d for mt, d in self.msgs if mt == t]

    # ----- dataspace
    @property
    def shape(self):
        d = self.msg(0x01)
        if d is None:
            return None
        ver = d[0]
        nd = d[1]
        if ver == 1:
            p = 8
        else:
            p = 4
            if d[3] == 2:
                return None
        return tuple(struct.unpack_from("<" + "Q" * nd, d, p)) if nd else ()

    @property
    def dtype_desc(self):
        d = self.msg(0x03)
        return parse_dtype(self.h5, d)[0] if d else None

    # ----- attributes
    def attrs(self):
        out = {}
        for d in self.msgs_of(0x0C):
            ver = d[0]
            if ver == 1:
                nsz, dtsz, dssz = struct.unpack_from("<HHH", d, 2)
                p = 8
                name = d[p:p + nsz].split(b"\x00")[0].decode()
                p += (nsz + 7) // 8 * 8
                dt = d[p:p + dtsz]
                p += (dtsz + 7) // 8 * 8
                ds = d[p:p + dssz]
                p += (dssz + 7) // 8 * 8
            else:
                off = 8 if ver == 3 else 6
                nsz, dtsz, dssz = struct.unpack_from("<HHH", d, 2)
                p = 8 if ver == 3 else 8
                name = d[p:p + nsz].split(b"\x00")[0].decode()
                p += nsz
                dt = d[p:p + dtsz]
                p += dtsz
                ds = d[p:p + dssz]
                p += dssz
            desc, _ = parse_dtype(self.h5, dt)
            if ds[0] == 1:
                nd = ds[1]
                shape = struct.unpack_from("<" + "Q" * nd, ds, 8) if nd else ()
            else:
                nd = ds[1]
                shape = struct.unpack_from("<" + "Q" * nd, ds, 4) if nd else ()
            n = int(np.prod(shape)) if shape else 1
            out[name] = self._decode(desc, d[p:], shape, n)
        return out

    # ----- decoding raw bytes -> python / numpy
    def _decode(self, desc, raw, shape, n):
        c = desc["class"]
        if c == 9:  # vlen
            res = []
            for i in range(n):
                ln, caddr, idx = struct.unpack_from("<IQI", raw, i * 16)
                blob = self.h5.gheap_obj(caddr, idx)[:ln * (desc["base"]["size"] if desc["vtype"] == 0 else 1)]
                if desc["vtype"] == 1:
                    res.append(blob.decode("utf-8", "replace"))
                else:
                    res.append(np.frombuffer(blob, np_dtype(desc["base"])))
            if not shape:
                return res[0]
            return np.array(res, dtype=object).reshape(shape)
        if c == 6 and any(m[2]["class"] == 9 for m in desc["members"]):
            # compound with vlen members: decode manually into list of dicts
            recs = []
            for i in range(n):
                base = i * desc["size"]
                rec = {}
                for name, moff, sub in desc["members"]:
                    if sub["class"] == 9:
                        ln, caddr, idx = struct.unpack_from("<IQI", raw, base + moff)
                        blob = self.h5.gheap_obj(caddr, idx)
                        rec[name] = blob.decode("utf-8", "replace") if sub["vtype"] == 1 else blob
                    else:
                        rec[name] = np.frombuffer(raw[base + moff: base + moff + sub["size"]], np_dtype(sub))[0]
                recs.append(rec)
            return recs
        dt = np_dtype(desc)
        arr = np.frombuffer(raw[: n * dt.itemsize], dtype=dt)
        if c == 3:
            arr = np.array([x.split(b"\x00")[0].decode("utf-8", "replace") for x in arr])
        arr = arr.reshape(shape) if shape else arr.reshape(())
        return arr[()] if not shape else arr

    # ----- dataset reading
    def read(self):
        shape = self.shape
        desc = self.dtype_desc
        n = int(np.prod(shape)) if shape else 1
        lay = self.msg(0x08)
        ver = lay[0]
        if ver == 3:
            cls = lay[1]
            if cls == 0:  # compact
                sz = struct.unpack_from("<H", lay, 2)[0]
                raw = lay[4:4 + sz]
                return self._decode(desc, raw, shape, n)
            if cls == 1:  # contiguous
                addr, size = struct.unpack_from("<QQ", lay, 2)
                raw = self.h5.b[addr: addr + size]
                return self._decode(desc, raw, shape, n)
            if cls == 2:  # chunked
                nd = lay[2]
                baddr = struct.unpack_from("<Q", lay, 3)[0]
                cdims = struct.unpack_from("<" + "I" * nd, lay, 11)
                return self._read_chunked(desc, shape, baddr, cdims[:-1])
        else:
            nd = lay[1]
            cls = lay[2]
            if cls == 1:
                addr = struct.unpack_from("<Q", lay, 8)[0]
                dt = np_dtype(desc)
                raw = self.h5.b[addr: addr + n * dt.itemsize]
                return self._decode(desc, raw, shape, n)
            if cls == 2:
                baddr = struct.unpack_from("<Q", lay, 8)[0]
                cdims = struct.unpack_from("<" + "I" * nd, lay, 16)
                return self._read_chunked(desc, shape, baddr, cdims[:-1])
        raise NotImplementedError(f"layout v{ver} class {lay[1]}")

    def _filters(self):
        d = self.msg(0x0B)
        if d is None:
            return []
        ver, nf = d[0], d[1]
        p = 8 if ver == 1 else 2
        fl = []
        for _ in range(nf):
            if ver == 1:
                fid, nlen, fflags, ncv = struct.unpack_from("<HHHH", d, p)
                p += 8
                p += (nlen + 7) // 8 * 8
                vals = struct.unpack_from("<" + "I" * ncv, d, p)
                p += 4 * ncv
                if ncv % 2:
                    p += 4
            else:
                fid = struct.unpack_from("<H", d, p)[0]
                p += 2
                nlen = 0
                if fid >= 256:
                    nlen = struct.unpack_from("<H", d, p)[0]
                    p += 2
                fflags, ncv = struct.unpack_from("<HH", d, p)
                p += 4
                p += nlen
                vals = struct.unpack_from("<" + "I" * ncv, d, p)
                p += 4 * ncv
            fl.append((fid, vals))
        return fl

    def _read_chunked(self, desc, shape, baddr, cdims):
        b = self.h5.b
        dt = np_dtype(desc)
        filters = self._filters()
        nd = len(shape)
        out = np.zeros(shape, dtype=dt)
        chunks = []

        def walk(addr):
            assert b[addr:addr + 4] == b"TREE"
            ntype, level, nused = struct.unpack_from("<BBH", b, addr + 4)
            p = addr + 24
            ksz = 8 + 8 * (nd + 1)
            for i in range(nused):
                csize, fmask = struct.unpack_from("<II", b, p)
                offs = struct.unpack_from("<" + "Q" * (nd + 1), b, p + 8)
                p += ksz
                child = struct.unpack_from("<Q", b, p)[0]
                p += 8
                if level > 0:
                    walk(child)
                else:
                    chunks.append((csize, fmask, offs[:nd], child))
        walk(baddr)
        for csize, fmask, offs, caddr in chunks:
            raw = b[caddr: caddr + csize]
            for idx in range(len(filters) - 1, -1, -1):
                fid, vals = filters[idx]
                if fmask & (1 << idx):
                    continue
                if fid == 1:
                    raw = zlib.decompress(raw)
                elif fid == 2:
                    es = vals[0] if vals else dt.itemsize
                    a = np.frombuffer(raw, np.uint8)
                    nel = len(a) // es
                    raw = a[: nel * es].reshape(es, nel).T.tobytes() + a[nel * es:].tobytes()
                elif fid == 3:
                    raw = raw[:-4]
                else:
                    raise NotImplementedError(f"filter {fid}")
            carr = np.frombuffer(raw, dtype=dt, count=int(np.prod(cdims))).reshape(cdims)
            sl_out = tuple(slice(o, min(o + c, s)) for o, c, s in zip(offs, cdims, shape))
            sl_in = tuple(slice(0, s.stop - s.start) for s in sl_out)
            out[sl_out] = carr[sl_in]
        return out
