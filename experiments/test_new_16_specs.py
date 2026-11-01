from typing import List, Tuple

EXPANDED_V4_SPECS = [
    {
        "name": "matplotlib_grouped_bar_norm",
        "slice": "7. Multi-Library & SciPy/Sklearn",
        "slot_options": [
            ["[v / max(1e-12, sum(vals)) for v in vals]", "vals", "[0.0 for _ in vals]", "[v * sum(vals) for v in vals]"],
            ["[round(x + width / 2.0, 4) for x in xs]", "xs", "[0.0 for _ in xs]", "[x - width for x in xs]"],
            ["max(shares) if shares else 0.0", "min(shares)", "0.0", "-1.0"],
            ["( [round(s, 4) for s in shares], centers, round(peak, 4) ) if len(shares) == len(xs) else ([], [], 0.0)", "( [round(s, 4) for s in shares], centers, round(peak, 4) )", "([], [], 0.0)", "None"],
        ],
        "template": (
            "\ndef solve(xs, vals, width=0.4):\n"
            "    shares = {s0}\n"
            "    centers = {s1}\n"
            "    peak = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([0.0, 1.0], [25.0, 75.0], width=0.4) == ([0.25, 0.75], [0.2, 1.2], 0.75)",
            "assert solve([1.0, 2.0, 3.0], [10.0, 10.0, 20.0], width=0.2) == ([0.25, 0.25, 0.5], [1.1, 2.1, 3.1], 0.5)",
            "assert solve([0.0], [10.0], width=0.5) == ([1.0], [0.25], 1.0)",
            "assert solve([0.0, 1.0], [10.0], width=0.4) == ([], [], 0.0)",
        ],
    },
    {
        "name": "scipy_ztest_two_proportions",
        "slice": "7. Multi-Library & SciPy/Sklearn",
        "slot_options": [
            ["(x1 + x2) / (n1 + n2)", "(x1 / n1 + x2 / n2)", "0.5", "0.0"],
            ["math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))", "p_pool * (1.0 - p_pool)", "1.0", "0.0"],
            ["(p1 - p2) / se", "p1 - p2", "(p1 + p2) / se", "0.0"],
            ["round(math.erfc(abs(z) / math.sqrt(2.0)), 4) if se > 0 else 1.0", "round(math.erf(abs(z)), 4)", "1.0", "0.0"],
        ],
        "template": (
            "\nimport math\n"
            "def solve(x1, n1, x2, n2):\n"
            "    p1, p2 = x1 / n1, x2 / n2\n"
            "    p_pool = {s0}\n"
            "    se = {s1}\n"
            "    z = {s2} if se > 0 else 0.0\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(50, 100, 50, 100) == 1.0",
            "assert solve(80, 100, 20, 100) == 0.0",
            "assert solve(60, 100, 45, 100) == 0.0337",
            "assert solve(0, 10, 0, 10) == 1.0",
        ],
    },
    {
        "name": "sklearn_stratified_fold_indices",
        "slice": "7. Multi-Library & SciPy/Sklearn",
        "slot_options": [
            ["buckets.setdefault(y, []).append(i)", "buckets.setdefault(0, []).append(i)", "pass", "buckets.clear()"],
            ["idx_list[pos % k_folds].append(sample_idx)", "idx_list[0].append(sample_idx)", "idx_list[-1].append(sample_idx)", "pass"],
            ["[sorted(fold) for fold in idx_list]", "idx_list", "[]", "[[]] * k_folds"],
            ["folds if all(len(f) > 0 for f in folds) else []", "folds", "[]", "folds[:1]"],
        ],
        "template": (
            "\ndef solve(labels, k_folds=2):\n"
            "    buckets = {}\n"
            "    for i, y in enumerate(labels):\n"
            "        {s0}\n"
            "    idx_list = [[] for _ in range(k_folds)]\n"
            "    for y in sorted(buckets.keys()):\n"
            "        for pos, sample_idx in enumerate(buckets[y]):\n"
            "            {s1}\n"
            "    folds = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([0, 0, 1, 1], k_folds=2) == [[0, 2], [1, 3]]",
            "assert solve([1, 0, 1, 0, 1, 0], k_folds=3) == [[0, 1], [2, 3], [4, 5]]",
            "assert solve([0, 1, 0, 1], k_folds=2) == [[0, 1], [2, 3]]",
            "assert solve([0], k_folds=2) == []",
        ],
    },
    {
        "name": "pandas_entity_latest_post_sort",
        "slice": "7. Multi-Library & SciPy/Sklearn",
        "slot_options": [
            ["sorted(rows, key=lambda r: r['ts'])", "rows", "sorted(rows, key=lambda r: -r['ts'])", "[]"],
            ["seen[r['user_id']] = r", "seen.setdefault(r['user_id'], r)", "seen.clear()", "pass"],
            ["sorted(dedup, key=lambda r: r['user_id'])", "dedup", "rows", "dedup[:1]"],
            ["[(r['user_id'], r['val']) for r in res]", "[]", "[(r['user_id'], 0) for r in res]", "[(r['user_id'], r['val']) for r in rows]"],
        ],
        "template": (
            "\ndef solve(rows):\n"
            "    ordered = {s0}\n"
            "    seen = {}\n"
            "    for r in ordered:\n"
            "        {s1}\n"
            "    dedup = list(seen.values())\n"
            "    res = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([{'user_id': 'u2', 'ts': 1, 'val': 10}, {'user_id': 'u1', 'ts': 2, 'val': 20}, {'user_id': 'u2', 'ts': 9, 'val': 30}]) == [('u1', 20), ('u2', 30)]",
            "assert solve([{'user_id': 'b', 'ts': 5, 'val': 50}, {'user_id': 'a', 'ts': 1, 'val': 10}]) == [('a', 10), ('b', 50)]",
            "assert solve([{'user_id': 'x', 'ts': 3, 'val': 30}, {'user_id': 'x', 'ts': 1, 'val': 10}]) == [('x', 30)]",
            "assert solve([]) == []",
        ],
    },
    {
        "name": "hmac_sha256_signature_verify",
        "slice": "8. Systems, Crypto & Regex",
        "slot_options": [
            ["hmac.new(key.encode('utf-8'), msg.encode('utf-8'), hashlib.sha256).hexdigest()", "hashlib.sha256(msg.encode('utf-8')).hexdigest()", "key", "''"],
            ["hmac.compare_digest(expected, sig)", "expected != sig", "False", "True"],
            ["expected[:16]", "expected", "''", "sig"],
            ["(ok, prefix) if len(key) > 0 else (False, '')", "(ok, prefix)", "(False, '')", "(True, '')"],
        ],
        "template": (
            "\nimport hashlib\nimport hmac\n"
            "def solve(key, msg, sig):\n"
            "    expected = {s0}\n"
            "    ok = {s1}\n"
            "    prefix = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "import hmac, hashlib; s = hmac.new(b'secret', b'hello', hashlib.sha256).hexdigest(); assert solve('secret', 'hello', s) == (True, s[:16])",
            "import hmac, hashlib; s = hmac.new(b'secret', b'hello', hashlib.sha256).hexdigest(); assert solve('secret', 'hello', 'bad') == (False, s[:16])",
            "assert solve('', 'hello', 'any') == (False, '')",
            "import hmac, hashlib; s = hmac.new(b'k', b'm', hashlib.sha256).hexdigest(); assert solve('k', 'm', s)[0] is True",
        ],
    },
    {
        "name": "regex_structured_log_latency",
        "slice": "8. Systems, Crypto & Regex",
        "slot_options": [
            ["re.compile(r'status=(\\d+)\\s+ms=(\\d+)')", "re.compile(r'(\\d+)')", "re.compile(r'ms=(\\d+)')", "re.compile(r'status=(\\d+)')"],
            ["int(m.group(1)) == 200", "int(m.group(1)) != 200", "False", "True"],
            ["ok_latencies.append(int(m.group(2)))", "ok_latencies.append(int(m.group(1)))", "pass", "ok_latencies.clear()"],
            ["round(sum(ok_latencies) / len(ok_latencies), 2) if ok_latencies else -1.0", "sum(ok_latencies)", "-1.0", "0.0"],
        ],
        "template": (
            "\nimport re\n"
            "def solve(lines):\n"
            "    pat = {s0}\n"
            "    ok_latencies = []\n"
            "    for line in lines:\n"
            "        m = pat.search(line)\n"
            "        if m and {s1}:\n"
            "            {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(['status=200 ms=10', 'status=500 ms=99', 'status=200 ms=20']) == 15.0",
            "assert solve(['status=404 ms=50']) == -1.0",
            "assert solve(['status=200 ms=7']) == 7.0",
            "assert solve([]) == -1.0",
        ],
    },
    {
        "name": "token_bucket_rate_limiter",
        "slice": "8. Systems, Crypto & Regex",
        "slot_options": [
            ["max(0.0, ts - last_ts) * rate", "(ts - last_ts)", "0.0", "rate"],
            ["min(float(capacity), tokens + refill)", "tokens + refill", "float(capacity)", "0.0"],
            ["tokens >= cost", "tokens > capacity", "False", "True"],
            ["(accepted, round(tokens, 4)) if capacity > 0 else ([], 0.0)", "(accepted, round(tokens, 4))", "([], 0.0)", "accepted"],
        ],
        "template": (
            "\ndef solve(events, capacity=10.0, rate=2.0):\n"
            "    tokens = float(capacity)\n"
            "    last_ts = 0.0\n"
            "    accepted = []\n"
            "    for ts, cost in events:\n"
            "        refill = {s0}\n"
            "        tokens = {s1}\n"
            "        last_ts = ts\n"
            "        if {s2}:\n"
            "            tokens -= cost\n"
            "            accepted.append(True)\n"
            "        else:\n"
            "            accepted.append(False)\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([(0.0, 6.0), (1.0, 6.0), (2.0, 4.0)], capacity=10.0, rate=2.0) == ([True, True, False], 2.0)",
            "assert solve([(0.0, 12.0)], capacity=10.0, rate=1.0) == ([False], 10.0)",
            "assert solve([(0.0, 5.0), (5.0, 10.0)], capacity=10.0, rate=2.0) == ([True, True], 0.0)",
            "assert solve([(1.0, 1.0)], capacity=0.0, rate=2.0) == ([], 0.0)",
        ],
    },
    {
        "name": "lru_cache_ordered_eviction",
        "slice": "8. Systems, Crypto & Regex",
        "slot_options": [
            ["cache.move_to_end(k)", "cache.pop(k)", "cache.clear()", "pass"],
            ["cache.move_to_end(k)", "cache.popitem(last=False)", "pass", "cache.clear()"],
            ["cache.popitem(last=False)", "cache.popitem(last=True)", "cache.clear()", "pass"],
            ["(hits, list(cache.keys())) if len(ops) >= 2 else (-1, [])", "(hits, list(cache.keys()))", "(0, [])", "hits"],
        ],
        "template": (
            "\nfrom collections import OrderedDict\n"
            "def solve(ops, cap=2):\n"
            "    cache = OrderedDict()\n"
            "    hits = 0\n"
            "    for op, k, v in ops:\n"
            "        if op == 'get':\n"
            "            if k in cache:\n"
            "                {s0}\n"
            "                hits += 1\n"
            "        else:\n"
            "            cache[k] = v\n"
            "            {s1}\n"
            "            if len(cache) > cap:\n"
            "                {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([('put','a',1), ('put','b',2), ('get','a',0), ('put','c',3), ('get','b',0)], cap=2) == (1, ['a', 'c'])",
            "assert solve([('put','x',1), ('put','y',2), ('put','x',9), ('put','z',3)], cap=2) == (0, ['x', 'z'])",
            "assert solve([('put','a',1), ('get','a',0), ('get','a',0)], cap=1) == (2, ['a'])",
            "assert solve([('put','a',1)], cap=2) == (-1, [])",
        ],
    },
    {
        "name": "kahn_cycle_checked_toposort",
        "slice": "9. Guarded Prefix-Trap Completions",
        "slot_options": [
            ["[i for i in range(n) if indeg[i] == 0]", "list(range(n))", "[]", "[0]"],
            ["indeg[nxt] -= 1", "indeg[nxt] += 1", "indeg[u] -= 1", "pass"],
            ["heapq.heappush(pq, nxt)", "pq.append(u)", "pass", "heapq.heappop(pq)"],
            ["order if len(order) == n else []", "order", "[]", "order[:n]"],
        ],
        "template": (
            "\nimport heapq\n"
            "def solve(n, edges):\n"
            "    adj = [[] for _ in range(n)]\n"
            "    indeg = [0] * n\n"
            "    for u, v in edges:\n"
            "        adj[u].append(v)\n"
            "        indeg[v] += 1\n"
            "    pq = {s0}\n"
            "    heapq.heapify(pq)\n"
            "    order = []\n"
            "    _s = 0\n"
            "    while pq and _s < 32:\n"
            "        _s += 1\n"
            "        u = heapq.heappop(pq)\n"
            "        order.append(u)\n"
            "        for nxt in adj[u]:\n"
            "            {s1}\n"
            "            if indeg[nxt] == 0:\n"
            "                {s2}\n"
            "    if _s >= 32:\n"
            "        raise RuntimeError('loop')\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(4, [(0, 1), (1, 2), (2, 0)]) == []",
            "assert solve(4, [(0, 2), (1, 2), (2, 3)]) == [0, 1, 2, 3]",
            "assert solve(3, [(2, 1), (1, 0)]) == [2, 1, 0]",
            "assert solve(2, [(0, 1), (1, 0)]) == []",
        ],
    },
    {
        "name": "bracket_depth_unclosed_guard",
        "slice": "9. Guarded Prefix-Trap Completions",
        "slot_options": [
            ["max(max_d, len(st))", "len(s)", "0", "max_d"],
            ["not st or st.pop() != pairs[ch]", "st and st.pop() == pairs[ch]", "False", "True"],
            ["return -1", "continue", "break", "pass"],
            ["max_d if (not st) else -1", "max_d", "-1", "0"],
        ],
        "template": (
            "\ndef solve(s):\n"
            "    pairs = {')': '(', ']': '[', '}': '{'}\n"
            "    st = []\n"
            "    max_d = 0\n"
            "    for ch in s:\n"
            "        if ch in '([{':\n"
            "            st.append(ch)\n"
            "            max_d = {s0}\n"
            "        elif ch in pairs:\n"
            "            if {s1}:\n"
            "                {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve('(([]){})') == 3",
            "assert solve('(((') == -1",
            "assert solve('())') == -1",
            "assert solve('') == 0",
        ],
    },
    {
        "name": "centered_svd_rank1_energy",
        "slice": "9. Guarded Prefix-Trap Completions",
        "slot_options": [
            ["X - np.mean(X, axis=0, keepdims=True)", "X + np.mean(X, axis=0, keepdims=True)", "X", "np.zeros_like(X)"],
            ["np.linalg.svd(Xc, full_matrices=False)", "(Xc, np.zeros(1), Xc)", "None", "(Xc, np.ones(1), Xc)"],
            ["Xc @ vh[0]", "X @ vh[0]", "np.zeros(X.shape[0])", "vh[0]"],
            ["np.round(np.abs(scores), 4).tolist()", "[]", "[0.0] * X.shape[0]", "np.round(scores, 4).tolist()"],
        ],
        "template": (
            "\nimport numpy as np\n"
            "def solve(mat):\n"
            "    X = np.asarray(mat, dtype=np.float64)\n"
            "    Xc = {s0}\n"
            "    u, s, vh = {s1}\n"
            "    scores = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "sc = solve([[100.0, 200.0], [101.0, 202.0], [99.0, 198.0]]); assert abs(sc[0]) < 1e-6 and sc[1] > 2.0",
            "sc = solve([[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]]); assert sc == [2.0, 0.0, 2.0]",
            "sc = solve([[-1.0, -1.0], [1.0, 1.0]]); assert abs(sc[0] - 1.4142) < 1e-3",
            "sc = solve([[5.0, 5.0], [5.0, 5.0]]); assert sc == [0.0, 0.0]",
        ],
    },
    {
        "name": "causal_window_anomaly_detector",
        "slice": "9. Guarded Prefix-Trap Completions",
        "slot_options": [
            ["xs[i - w : i]", "xs[:i + 1]", "xs", "xs[i : i + w]"],
            ["sum(win) / len(win)", "0.0", "max(win)", "min(win)"],
            ["(sum((v - mu) ** 2 for v in win) / len(win)) ** 0.5", "1.0", "0.0", "sum(win)"],
            ["out if len(xs) >= w else []", "[]", "[0]", "list(range(len(xs)))"],
        ],
        "template": (
            "\ndef solve(xs, w=4, thresh=2.0):\n"
            "    out = []\n"
            "    for i in range(w, len(xs)):\n"
            "        win = {s0}\n"
            "        mu = {s1}\n"
            "        sd = {s2}\n"
            "        if abs(xs[i] - mu) / max(sd, 1e-8) > thresh:\n"
            "            out.append(i)\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([10.0, 14.0, 10.0, 14.0, 15.0, 19.0], w=4, thresh=2.1) == [5]",
            "assert solve([10.0, 14.0, 10.0, 14.0, 5.0], w=4, thresh=2.1) == [4]",
            "assert solve([2.0, 2.0, 2.0, 2.0, 2.0], w=4, thresh=2.0) == []",
            "assert solve([1.0, 2.0], w=4, thresh=2.0) == []",
        ],
    },
    {
        "name": "crt_coprime_system_solver",
        "slice": "10. Symbolic Math & Combinatorics",
        "slot_options": [
            ["math.prod(moduli)", "sum(moduli)", "max(moduli)", "1"],
            ["M // m_i", "M * m_i", "m_i", "1"],
            ["pow(M_i, -1, m_i)", "pow(m_i, -1, M_i)", "1", "M_i % m_i"],
            ["total % M", "total", "M", "0"],
        ],
        "template": (
            "\nimport math\n"
            "def solve(remainders, moduli):\n"
            "    M = {s0}\n"
            "    total = 0\n"
            "    for r_i, m_i in zip(remainders, moduli):\n"
            "        M_i = {s1}\n"
            "        inv_i = {s2}\n"
            "        total += r_i * M_i * inv_i\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve([2, 3, 2], [3, 5, 7]) == 23",
            "assert solve([1, 3], [5, 7]) == 31",
            "assert solve([0, 0], [5, 7]) == 0",
            "assert solve([3, 4], [4, 5]) == 19",
        ],
    },
    {
        "name": "lucas_theorem_binomial_mod_p",
        "slice": "10. Symbolic Math & Combinatorics",
        "slot_options": [
            ["n % p, k % p", "n // p, k // p", "n, k", "0, 0"],
            ["return 0", "continue", "break", "pass"],
            ["(ans * math.comb(ni, ki)) % p", "ans + math.comb(ni, ki)", "math.comb(ni, ki)", "0"],
            ["ans % p", "0", "1", "p"],
        ],
        "template": (
            "\nimport math\n"
            "def solve(n, k, p):\n"
            "    if k < 0 or k > n:\n"
            "        return 0\n"
            "    ans = 1\n"
            "    while n > 0 or k > 0:\n"
            "        ni, ki = {s0}\n"
            "        if ki > ni:\n"
            "            {s1}\n"
            "        ans = {s2}\n"
            "        n //= p\n"
            "        k //= p\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(1000, 900, 13) == 8",
            "assert solve(10, 2, 7) == 3",
            "assert solve(7, 3, 5) == 0",
            "assert solve(5, 0, 5) == 1",
        ],
    },
    {
        "name": "euler_totient_factorization",
        "slice": "10. Symbolic Math & Combinatorics",
        "slot_options": [
            ["ans - ans // d", "ans // d", "ans - d", "ans"],
            ["temp //= d", "temp -= d", "break", "pass"],
            ["ans - ans // temp", "ans // temp", "ans", "0"],
            ["ans if n > 0 else 0", "temp", "0", "n"],
        ],
        "template": (
            "\ndef solve(n):\n"
            "    if n <= 0:\n"
            "        return 0\n"
            "    ans = n\n"
            "    temp = n\n"
            "    d = 2\n"
            "    while d * d <= temp:\n"
            "        if temp % d == 0:\n"
            "            _s = 0\n"
            "            while temp % d == 0 and _s < 32:\n"
            "                _s += 1\n"
            "                {s1}\n"
            "            if _s >= 32:\n"
            "                raise RuntimeError('loop')\n"
            "            ans = {s0}\n"
            "        d += 1\n"
            "    if temp > 1:\n"
            "        ans = {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(36) == 12",
            "assert solve(97) == 96",
            "assert solve(1) == 1",
            "assert solve(100) == 40",
        ],
    },
    {
        "name": "inclusion_exclusion_multiples",
        "slice": "10. Symbolic Math & Combinatorics",
        "slot_options": [
            ["math.lcm(cur_lcm, primes[b])", "cur_lcm + primes[b]", "primes[b]", "1"],
            ["1 if (bits % 2 == 1) else -1", "-1 if (bits % 2 == 1) else 1", "1", "-1"],
            ["sign * (limit // cur_lcm)", "limit // cur_lcm", "sign * cur_lcm", "0"],
            ["total if limit >= 1 else 0", "0", "limit", "-total"],
        ],
        "template": (
            "\nimport math\n"
            "def solve(limit, primes):\n"
            "    m = len(primes)\n"
            "    total = 0\n"
            "    for mask in range(1, 1 << m):\n"
            "        cur_lcm = 1\n"
            "        bits = 0\n"
            "        for b in range(m):\n"
            "            if (mask >> b) & 1:\n"
            "                cur_lcm = {s0}\n"
            "                bits += 1\n"
            "        sign = {s1}\n"
            "        total += {s2}\n"
            "    return {s3}\n"
        ),
        "tests": [
            "assert solve(100, [2, 3, 5]) == 74",
            "assert solve(20, [2, 3]) == 13",
            "assert solve(10, [7]) == 1",
            "assert solve(0, [2, 3]) == 0",
        ],
    },
]

def render(spec, choices):
    code = spec["template"]
    for _ in range(4):
        for k, c in enumerate(choices):
            code = code.replace(f"{{s{k}}}", spec["slot_options"][k][c])
    return code

if __name__ == "__main__":
    for spec in EXPANDED_V4_SPECS:
        code0 = render(spec, [0, 0, 0, 0])
        ns = {}
        exec(code0, ns, ns)
        for t in spec["tests"]:
            try:
                exec(t, ns, ns)
            except Exception as e:
                print(f"FAILED ground truth on {spec['name']}: {t} -> {e}")
                raise
        # Check that each wrong choice fails at least one test
        for k in range(4):
            for c in range(1, 4):
                ch = [0, 0, 0, 0]
                ch[k] = c
                code_bad = render(spec, ch)
                failed = False
                try:
                    ns_b = {}
                    exec(code_bad, ns_b, ns_b)
                    for t in spec["tests"]:
                        exec(t, ns_b, ns_b)
                except Exception:
                    failed = True
                assert failed, f"Spec {spec['name']} slot {k} option {c} did NOT fail any test!"

    print(f"ALL {len(EXPANDED_V4_SPECS)} EXPANDED_V4_SPECS (64 slots, 256 candidates) VERIFIED 100% DISCRIMINATIVE!")
