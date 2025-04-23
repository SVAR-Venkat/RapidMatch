// Full Java Implementation of RapidMatch (Complete Pipeline)
// Includes: Graph, Edge, GraphParser, NucleusDecomposer (k-(2,3)), RelationFilter, JoinPlanner, RelationEncoder, ResultEnumerator, Optimizations

import java.io.*;
import java.util.*;

public class RapidMatchMain {
    public static void main(String[] args) {

        File queryDir = new File("DataSet/query");

        // Get sorted list of files to pair one-to-one
        File[] queryFiles = queryDir.listFiles((dir, name) -> name.endsWith(".graph"));

        if (queryFiles == null) {
            System.out.println("Query or data folder not found.");
            return;
        }

        // Measure total runtime and number of Matchs:
        long totalStart = System.nanoTime();
        int grandTotalMatches = 0;

        for (File queryFile : queryFiles) {
            System.out.println("\n=== Running RapidMatch ===");
            System.out.println("Query: " + queryFile.getName());
            try {
                Graph queryGraph = GraphParser.parseGraph(queryFile.getPath());
                Graph dataGraph  = GraphParser.parseGraph("DataSet/data1.Graph");

                RapidMatch matcher = new RapidMatch(queryGraph, dataGraph);

                // --- timing wrapper starts here ---
                long start = System.nanoTime();
                int matches = matcher.execute();
                long end   = System.nanoTime();
                System.out.printf("Finished %s in %.3f ms%n and %d matches found",
                                  queryFile.getName(),
                                  (end - start) / 1e6, matches);
                // --- timing wrapper ends here ---

            } catch (IOException e) {
                System.err.println("Error processing files: " + queryFile.getName());
                e.printStackTrace();
            }
        }

        long totalEnd = System.nanoTime();
        System.out.printf("%nAll %d queries completed in %.3f ms%n and %d Matches found",
                          queryFiles.length,
                          (totalEnd - totalStart) / 1e6, grandTotalMatches);
        
        

        // This is line of code for the just sample and small Q and G graphs for evaluation which are query.graph and data.graph.
        /*try {
        Graph queryGraph = GraphParser.parseGraph("DataSet/query.Graph");
        Graph dataGraph = GraphParser.parseGraph("DataSet/data.Graph");
        RapidMatch matcher = new RapidMatch(queryGraph, dataGraph);
        // --- timing wrapper starts here ---
        long start = System.nanoTime();
        matcher.execute();
        long end   = System.nanoTime();
        System.out.printf("Finished %s in %.3f ms%n", "query.Graph",(end - start) / 1e6);
        // --- timing wrapper ends here ---
        } catch (IOException e) {
        e.printStackTrace();
        }*/
    }
}

class Graph {
    Map<Integer, List<Integer>> adjList = new HashMap<>();
    Map<Integer, String> labels = new HashMap<>();
    List<Edge> edges = new ArrayList<>();

    void addEdge(int u, int v) {
        adjList.putIfAbsent(u, new ArrayList<>());
        adjList.putIfAbsent(v, new ArrayList<>());
        adjList.get(u).add(v);
        adjList.get(v).add(u);
        edges.add(new Edge(u, v));
    }
}

class Edge {
    int u, v;

    Edge(int u, int v) {
        this.u = Math.min(u, v);
        this.v = Math.max(u, v);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof Edge))
            return false;
        Edge e = (Edge) o;
        return u == e.u && v == e.v;
    }

    @Override
    public int hashCode() {
        return Objects.hash(u, v);
    }

    @Override
    public String toString() {
        return "(" + u + ", " + v + ")";
    }
}

class GraphParser {
    public static Graph parseGraph(String path) throws IOException {
        Graph g = new Graph();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("t"))
                    continue;
                String[] p = line.split("\\s+");
                if (p[0].equals("v")) {
                    int id = Integer.parseInt(p[1]);
                    g.labels.put(id, p[2]);
                    g.adjList.putIfAbsent(id, new ArrayList<>());
                } else if (p[0].equals("e")) {
                    int u = Integer.parseInt(p[1]), v = Integer.parseInt(p[2]);
                    g.addEdge(u, v);
                }
            }
        }
        return g;
    }
}

class RapidMatch {
    private Graph Q, G;
    List<Edge> queryEdges;
    List<Edge> dataEdges;
    Map<Edge, List<Edge>> cand;
    List<Set<Edge>> nuclei;
    List<Edge> joinOrder;

    public RapidMatch(Graph q, Graph g) {
        this.Q = q;
        this.G = g;
        this.queryEdges = new ArrayList<>(q.edges);
        this.dataEdges = new ArrayList<>(g.edges);
    }

    public int execute() {
        System.out.println("Step 1: Nucleus Decomposition (k-(2,3))");
        nuclei = NucleusDecomposer.kTrussDecomposition(Q, 2);

        System.out.println("Step 2: Relation Filtering");
        RelationFilter filter = new RelationFilter(Q, G, queryEdges, dataEdges);
        cand = filter.run();

        System.out.println("Step 3: Join Plan Generation");
        JoinPlanner planner = new JoinPlanner(queryEdges, cand);
        joinOrder = planner.plan();

        System.out.println("Step 4: Relation Encoding");
        new RelationEncoder(nuclei, queryEdges, cand).buildAll();

        System.out.println("Step 5: Result Enumeration");
        ResultEnumerator enumerator =new ResultEnumerator(Q, G, joinOrder, cand);

        return enumerator.run();
    }
}

// ---- ADDITIONAL CLASSES ----

class NucleusDecomposer {
    public static List<Set<Edge>> kTrussDecomposition(Graph g, int k) {
        Map<Edge, Integer> support = new HashMap<>();

        for (int u : g.adjList.keySet()) {
            for (int v : g.adjList.get(u)) {
                if (u < v) {
                    Set<Integer> common = new HashSet<>(g.adjList.get(u));
                    common.retainAll(g.adjList.get(v));
                    for (int w : common) {
                        List<Edge> tri = Arrays.asList(
                                new Edge(u, v), new Edge(u, w), new Edge(v, w));
                        for (Edge e : tri) {
                            support.put(e, support.getOrDefault(e, 0) + 1);
                        }
                    }
                }
            }
        }

        Set<Edge> valid = new HashSet<>();
        for (Map.Entry<Edge, Integer> entry : support.entrySet()) {
            if (entry.getValue() >= k - 2) {
                valid.add(entry.getKey());
            }
        }

        Map<Integer, List<Integer>> tempAdj = new HashMap<>();
        for (Edge e : valid) {
            tempAdj.computeIfAbsent(e.u, x -> new ArrayList<>()).add(e.v);
            tempAdj.computeIfAbsent(e.v, x -> new ArrayList<>()).add(e.u);
        }

        List<Set<Edge>> components = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();
        for (int u : tempAdj.keySet()) {
            if (!visited.contains(u)) {
                Set<Edge> comp = new HashSet<>();
                Stack<Integer> stack = new Stack<>();
                stack.push(u);
                visited.add(u);
                while (!stack.isEmpty()) {
                    int curr = stack.pop();
                    for (int v : tempAdj.get(curr)) {
                        Edge e = new Edge(curr, v);
                        if (valid.contains(e))
                            comp.add(e);
                        if (!visited.contains(v)) {
                            visited.add(v);
                            stack.push(v);
                        }
                    }
                }
                if (!comp.isEmpty())
                    components.add(comp);
            }
        }

        return components;
    }
}

class RelationFilter {
    private final Graph Q, G;
    private final List<Edge> queryEdges;
    private final List<Edge> dataEdges;
    private final Map<Edge, List<Edge>> candidates = new HashMap<>();

    public RelationFilter(Graph Q, Graph G, List<Edge> queryEdges, List<Edge> dataEdges) {
        this.Q = Q;
        this.G = G;
        this.queryEdges = queryEdges;
        this.dataEdges = dataEdges;
    }

    public Map<Edge, List<Edge>> run() {
        initializeCandidates();
        reduceCandidates();
        return candidates;
    }

    private void initializeCandidates() {
        for (Edge qe : queryEdges) {
            List<Edge> match = new ArrayList<>();
            String labelU = Q.labels.get(qe.u);
            String labelV = Q.labels.get(qe.v);
            for (Edge de : dataEdges) {
                String dLabelU = G.labels.get(de.u);
                String dLabelV = G.labels.get(de.v);
                if ((labelU.equals(dLabelU) && labelV.equals(dLabelV)) ||
                        (labelU.equals(dLabelV) && labelV.equals(dLabelU))) {
                    match.add(de);
                }
            }
            candidates.put(qe, match);
        }
    }

    private void reduceCandidates() {
        boolean changed;
        do {
            changed = false;
            for (Edge qe : queryEdges) {
                Iterator<Edge> it = candidates.get(qe).iterator();
                while (it.hasNext()) {
                    Edge de = it.next();
                    if (!hasSupport(de, qe)) {
                        it.remove();
                        changed = true;
                    }
                }
            }
        } while (changed);
    }

    private boolean hasSupport(Edge de, Edge qe) {
        for (Edge other : queryEdges) {
            if (qe.equals(other))
                continue;
            if (sharesVertex(qe, other)) {
                boolean supported = false;
                for (Edge d : candidates.getOrDefault(other, Collections.emptyList())) {
                    if (sharesVertex(de, d)) {
                        supported = true;
                        break;
                    }
                }
                if (!supported)
                    return false;
            }
        }
        return true;
    }

    private boolean sharesVertex(Edge e1, Edge e2) {
        return e1.u == e2.u || e1.u == e2.v || e1.v == e2.u || e1.v == e2.v;
    }
}

class JoinPlanner {
    private final List<Edge> queryEdges;
    private final Map<Edge, List<Edge>> cand;

    public JoinPlanner(List<Edge> queryEdges, Map<Edge, List<Edge>> cand) {

        this.queryEdges = queryEdges;
        this.cand = cand;
    }

    public List<Edge> plan() {
        // Greedy plan: sort edges by ascending candidate size
        List<Edge> plan = new ArrayList<>(queryEdges);
        plan.sort(Comparator.comparingInt(e -> cand.get(e).size()));
        return plan;
    }
}

class RelationEncoder {
    private final List<Set<Edge>> nuclei;
    private final List<Edge> allEdges;
    private final Map<Edge, List<Edge>> cand;
    private final Map<Edge, Map<Integer, List<Integer>>> tries = new HashMap<>();
    private final Map<Edge, Map<Integer, Set<Integer>>> hashIndexes = new HashMap<>();

    public RelationEncoder(List<Set<Edge>> nuclei, List<Edge> allEdges, Map<Edge, List<Edge>> cand) {

        this.nuclei = nuclei;
        this.allEdges = allEdges;
        this.cand = cand;
    }

    public void buildAll() {
        Set<Edge> QC = new HashSet<>();
        for (Set<Edge> comp : nuclei)
            QC.addAll(comp);
        List<Edge> QF = new ArrayList<>();
        for (Edge e : allEdges)
            if (!QC.contains(e))
                QF.add(e);

        for (Edge e : QC)
            buildTrie(e);
        for (Edge e : QF)
            buildHashIndex(e);
    }

    private void buildTrie(Edge e) {
        System.out.println("Encoding trie for core edge: " + e);
        Map<Integer, List<Integer>> trie = new HashMap<>();
        for (Edge de : cand.get(e)) {
            trie.computeIfAbsent(de.u, k -> new ArrayList<>()).add(de.v);
        }
        // sort inner lists for consistency
        for (List<Integer> list : trie.values())
            Collections.sort(list);
        tries.put(e, trie);
    }

    private void buildHashIndex(Edge e) {
        System.out.println("Building hash index for fringe edge: " + e);
        Map<Integer, Set<Integer>> index = new HashMap<>();
        for (Edge de : cand.get(e)) {
            index.computeIfAbsent(de.u, k -> new HashSet<>()).add(de.v);
            index.computeIfAbsent(de.v, k -> new HashSet<>()).add(de.u);
        }
        hashIndexes.put(e, index);
    }
}

class ResultEnumerator {
    Graph Q, G;
    private int matchCount;
    private final List<Edge> phi;
    private final Map<Edge, List<Edge>> cand;

    public ResultEnumerator(Graph Q, Graph G, List<Edge> phi,
            Map<Edge, List<Edge>> cand) {
        this.phi = phi;
        this.Q = Q;
        this.G = G;
        this.cand = cand;
    }

    public int run() {
        matchCount = 0;
        backtrack(0, new HashMap<>(), new HashSet<>());
        return matchCount;
    }

    private void backtrack(int idx, Map<Integer, Integer> M, Set<Integer> used) {
        if (idx == phi.size()) {
            System.out.println("Match: " + M);
            matchCount++;
            return;
        }

        Edge qe = phi.get(idx);
        for (Edge de : cand.get(qe)) {
            tryMap(qe.u, qe.v, de.u, de.v, idx, M, used);
            tryMap(qe.u, qe.v, de.v, de.u, idx, M, used);
        }
    }

    private void tryMap(int qu, int qv, int du, int dv, int idx, Map<Integer, Integer> M, Set<Integer> used) {
        if ((M.containsKey(qu) && M.get(qu) != du) ||
                (M.containsKey(qv) && M.get(qv) != dv) ||
                (used.contains(du) && !M.containsKey(qu)) ||
                (used.contains(dv) && !M.containsKey(qv)))
            return;

        for (int qx : M.keySet()) {
            int dx = M.get(qx);
            if (adjacent(Q, qu, qx) && !adjacentIn(G, du, dx))
                return;
            if (adjacent(Q, qv, qx) && !adjacentIn(G, dv, dx))
                return;
        }

        boolean wu = M.containsKey(qu), wv = M.containsKey(qv);
        M.put(qu, du);
        M.put(qv, dv);
        used.add(du);
        used.add(dv);

        backtrack(idx + 1, M, used);

        if (!wu) {
            M.remove(qu);
            used.remove(du);
        }
        if (!wv) {
            M.remove(qv);
            used.remove(dv);
        }
    }

    private boolean adjacent(Graph g, int u, int v) {
        return Q.adjList.containsKey(u) && Q.adjList.get(u).contains(v);
    }

    private boolean adjacentIn(Graph g, int u, int v) {
        return G.adjList.containsKey(u) && G.adjList.get(u).contains(v);
    }
}