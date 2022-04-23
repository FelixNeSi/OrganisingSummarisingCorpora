from preprocessing import *
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import itertools


def create_transformer_model(model_name='distilbert-base-nli-mean-tokens'):
    model = SentenceTransformer(model_name)
    return model


def encode_corpus_with_model(data, model):
    embeddings = model.encode(data, show_progress_bar=True, convert_to_tensor=True)
    return embeddings


def kmeans_cluster(corpus_embeddings, num_clusters=5):
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def aglo_cluster(corpus_embeddings, n_clusters=5, distance_threshold=None):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters,
                                               distance_threshold=distance_threshold)  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    return cluster_assignment


def fast_cluster(corpus_embeddings, min_community_size=25, threshold=0.75):
    clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold,
                                        init_max_size=len(corpus_embeddings))
    assignments = [None for x in range(len(corpus_embeddings))]
    for i, cluster in enumerate(clusters):
        for doc in cluster:
            assignments[doc] = i
    # print(assignments)
    return assignments


def print_clusters(cluster_assignment, corpus):
    clustered_sentences = []
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")


def normalise_embeddings(embeddings):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


def generate_candidate_keywords(doc, min_ngram=1, max_ngram=1, remove_stopwords=True):
    n_gram_range = (min_ngram, max_ngram)
    stop_words = "english"

    # Extract candidate words/phrases
    if remove_stopwords:
        count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit(doc)
    else:
        count = CountVectorizer(ngram_range=n_gram_range).fit(doc)
    candidates = count.get_feature_names()
    return candidates


def get_most_cosine_similar(doc_embedding, candidate_embeddings, candidates, top_n=5):
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings,
                                             candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]


def mmr(doc_embedding, word_embeddings, candidates, top_n, diversity):
    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [candidates[idx] for idx in keywords_idx]


def do_bert_keyword_extraction(data, sim_method="cosine", top_n=5, nr_candidates=10, diversity=0.2):
    candidates = [generate_candidate_keywords([doc]) for doc in data]
    model = create_transformer_model()
    doc_embeddings = [model.encode([doc]) for doc in data]
    candidate_embeddings = [model.encode(cand) for cand in candidates]
    keywords = []
    for i in range(len(data)):
        if sim_method == "cosine":
            keywords.append(
                get_most_cosine_similar(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n=top_n))
        elif sim_method == "max_sum":
            keywords.append(
                max_sum_sim(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n, nr_candidates))
        elif sim_method == "max_marginal":
            keywords.append(mmr(doc_embeddings[i], candidate_embeddings[i], candidates[i], top_n, diversity))
    return keywords
    # if sim_method == "cosine":
    #     get_most_cosine_similar()
    # print(candidates)
    # print(len(doc_embeddings))
    # print(len(candidate_embeddings))
    # print(len(candidates))

# data = ["This is a 1 TEST sentence that contains CaSes CaSes CaSes CaSes",
#         "CaSes CaSes CaSes CaSes ALL OF THE STOPWORDS !@!@!@@", "CaSes CaSes Good TESTS go For EdGe CaSes"
#     , "Good TESTS go For EdGe CaSes CaSes CaSes CaSes", "Good TESTS go For EdGe CaSes CaSes CaSes CaSes"]
# #
# # # ['cases', 'contains', 'edge', 'good', 'sentence', 'stopwords', 'test', 'tests']
# #
# # # print(generate_candidate_keywords(data[0]))
# # print(do_bert_keyword_extraction(data))
# model = create_transformer_model('distilbert-base-nli-mean-tokens')
# embeddings = encode_corpus_with_model(data, model)
# # embeddings = normalise_embeddings(embeddings)
# assignments = kmeans_cluster(embeddings, 3)
# # for i, cluster in enumerate(clusters):
# #     print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
# #     for sentence_id in cluster[0:3]:
# #         print("\t", corpus_sentences[sentence_id])
# #     print("\t", "...")
# #     for sentence_id in cluster[-3:]:
# #         print("\t", corpus_sentences[sentence_id])
# print(assignments)
# print_clusters(assignments, data)
#
# # print(embeddings)
#
# test = [['temporal logic', 'concurrent protocol verification', 'explicit enumeration', 'in-stack checking', 'on-the-fly model-checking', 'partial order reductions', 'proviso'], ['fault-tolerance', 'formal methods', 'dependability', 'graceful degradation', 'correctors', 'detectors', 'compositional design', 'interference-freedom', 'stepwise design'], ['distributed computing', 'concurrency control', 'consistency maintenance', 'collaborative editing', 'graphics editing'], ['flexible coupling', 'merging', 'optimistic concurrency control', 'versions'], ['heavy traffic', 'multiclass queueing network', 'fluid approximation', 'diffusion approximation', 'priority service discipline', 'semimartingale reflecting Brownian motion'], ['load balancing', 'large deviations', 'queuing theory', 'stale information', 'old information'], ['distributed systems', 'load balancing', 'server selection', 'queuing theory', 'stale information'], ['real-time', 'scheduling', 'task model', 'utilization bound'], ['debugging', 'race conditions', 'parallel programs', 'critical sections', 'data races', 'nondeterminacy'], ['nested transactions', 'concurrency control', 'locking', 'object hierarchies'], ['distributed algorithms', 'randomized algorithms', 'multiparty interaction', 'strong interaction fairness', 'weak interaction fairness', 'committee coordination'], ['multithreading', 'garbage collection', 'programming language implementation'], ['multiprocessors', 'memory management', 'competitive analysis', 'on-line algorithms', 'page migration'], ['continuations', 'language design', 'multiprocessing', 'operating sytems', 'scheme'], ['makespan', 'stochastic scheduling', 'in-forest', 'interval order', 'out-forest', 'precedence constraint', 'profile scheduling', 'stochastic ordering', 'uniform processors'], ['distributed processing', 'Esprit Delta-4 Project', 'dependability evaluation', 'dependability measures', 'distributed fault-tolerant architecture', 'fault injection', 'fault occurrence process', 'fault tolerance process', 'fault tolerant computing', 'fault-tolerant systems', 'test sequence'], ['competitive analysis', 'online bipontite matching', 'online scheduling'], ['wait-free algorithms', 'asynchronous shared-memory systems', 'contention-sensitive algorithms', 'deterministic coin tossing', 'load-linked/store-conditional operations', 'universal operations'], ['real-time', 'constraint solving', 'end-to-end timing constraints', 'design methodology', 'static priority scheduling', 'non-linear optimization'], ['optimization', 'data structures', 'discrete event simulation', 'markov chain', 'priority queue', 'algorithm analysis', 'calendar queue'], ['stability', 'acyclic routing', 'fluid model', 'multiclass queueing network', 'push start', 'virtual station'], ['replicated databases', 'availability', 'quorum consensus', 'replica control', 'message overhead', 'update synchronization'], ['distributed databases', 'fault tolerant computing', 'concurrency control', 'distributed environment', 'petri nets', 'file status', 'majority protocols', 'performance reliability tradeoffs', 'replicated file system', 'stochastic Petri net model', 'voting algorithm', 'witnesses'], ['self-stabilization', 'message passing', 'shared memory', 'token passing'], ['trees mathematics', 'index termsdistributed system', 'composite coterie', 'distributedalgorithms', 'join algorithm', 'nonempty coteries', 'nonempty intersection', 'read and write quorums', 'replica control protocol', 'treecoteries'], ['shared memory', 'cache coherence', 'Lazy Release Consistency', 'protocol processors'], ['error-correcting codes', 'communication complexity', 'NMR system', 'majority voting', 'MDS code'], ['fault tolerant computing', 'simulation results', 'digital simulation', 'processing modules', 'redundancy', 'bayes methods', 'bayes theorem', 'TMR failures', 'adaptive recovery method', 'disagreement detector', 'fault-state likelihoods', 'system reconfiguration', 'time redundancy approach', 'triple modular redundant system', 'voters'], ['storage management', 'garbage collection', 'distributed object systems', 'reference tracking'], ['distributed processing', 'fault tolerance', 'leader election', 'fault tolerant computing', 'reliability', 'robustness', 'system recovery', 'embedded system', 'channel failures', 'channel repairs', 'diffusing computation', 'distributed reset subsystem', 'fail-stop failure tolerance', 'layered design', 'process failures', 'process repairs', 'self-stabilizing components', 'spanning tree construction', 'up-process coordination'], ['concurrency', 'wait-free', 'lock-free', 'shared objects', 'nonblocking synchronization'], ['condensed graphs', 'functional and dataflow programming', 'imperative', 'protection mechanisms', 'security models'], ['distributed systems', 'load sharing', 'sharable jobs', 'job sharing coefficient'], ['memory management', 'competitive analysis', 'on-line algorithms', 'data management'], ['distributed algorithms', 'distributed debugging', 'predicate detection', 'unstable predicates'], ['garbage collection', 'scalability', 'parallel algorithm', 'dynamic load balancing', 'shared-memory machine'], ['distributed systems', 'distributed algorithms', 'clock synchronization', 'TCP/IP', 'local area networks', 'probabilistic algorithms'], ['stochastic modeling', 'queueing systems', 'transient loss performance'], ['on-line algorithms', 'searching', 'bayesian models', 'randomized adversary'], ['protocol verification', 'cache coherence', 'memory consistency', 'protocol specification', 'multicast snooping'], ['programming languages', 'concurrency', 'semantics', 'ethernet', 'broadcasting', 'calculi'], ['distributed systems', 'asynchrony', 'group communication', 'sliding windows', 'causal broadcasting', 'vector times', 'happened-before relation'], ['scheduling', 'CHARMS', 'partial', 'progressive', 'result extraction'], ['parallel processing', 'mutual exclusion', 'priority queue', 'real-time system', 'spin lock'], ['fault tolerance', 'replication', 'byzantine failures', 'quorum systems'], ['fault-containment', 'mutual exclusion', 'self-stabilization'], ['fault-tolerance', 'randomized algorithms', 'shared memory', 'atomicity', 'adaptive adversary', 'asynchronous distributed protocols', 'naming problem', 'symmetry breaking', 'test-and-set objects', 'unique process ID', 'wait-free read/write registers'], ['scheduling', 'approximation algorithms', 'real-time scheduling', 'multiple machines scheduling', 'parallel machines scheduling', 'throughput'], ['rate monotonic scheduling', 'deadline monotonic scheduling', 'feasibility decision algorithm', 'periodic task', 'priority inheritance protocol'], ['fault tolerance', 'checkpointing', 'rollback recovery', 'memory redundancy', 'error-correcting codes', 'copy-on-write', 'RAID systems'], ['checkpointing', 'rollback recovery', 'on-line algorithms', 'rollback-dependency trackability', 'communication-induced protocols'], ['optimistic concurrency control', 'groupware', 'model-view-controller programming paradigm', 'replicated objects', 'optimistic views', 'pessimistic views'], ['atomic read/write registers', 'renaming', 'shared-memory systems', 'wait-free computation', 'atomic snapshots', 'lattice agreement'], ['memory management', 'type theory', 'data representation', 'ordered logic'], ['shared-memory multiprocessors', 'formal methods', 'cache coherence protocols', 'state abstraction', 'state enumeration methods'], ['performance', 'garbage collection', 'uniprocessor'], ['heap cells', 'maximal matching', 'stack accesses', 'stack frames'], ['mobile agent', 'diffuse computation', 'finite-state agent', 'intermittent communication', 'sensor net', 'stable computation'], ['fault tolerance', 'replicated file', 'performance analysis', 'signatures', 'coding theory', 'file comparison', 'message transmission'], ['lower bound', 'distributed algorithm', 'fail-stop failures', 'multiple-access channel', 'adversary', 'independent tasks'], ['virtual time', 'time warp', 'logical time', 'vector clocks'], ['fault-tolerance', 'asynchronous distributed system', 'checkpointing protocols'], ['load balancing', 'checkpointing', 'backup process', 'fault-tolerant multicomputer', 'process allocation'], ['heavy traffic', 'queues', 'large deviation principle'], ['fault tolerance', 'real-time systems', 'periodic tasks', 'rate monotonic scheduling', 'minimum achievable utilization'], ['concurrency control', 'distributed file systems', 'lock evaluation', 'session locking'], ['communication protocols', 'refinement', 'DSM protocols'], ['static analysis', 'interprocedural analysis', 'memory preallocation'], ['parallel simulation', 'synchronization protocol'], ['distributed systems', 'accuracy', 'scalability', 'efficiency', 'failure detectors'], ['memory management', 'time warp simulation', 'anti-messages', 'dependency information', 'distributed recovery', 'logical process', 'message tagging', 'optimistic distributed simulation', 'optimistic distributed simulation protocols', 'process rollback', 'rollback broadcasting', 'straggler', 'transitive dependency information', 'transitive dependency tracking'], ['distributed systems', 'failure model', 'asynchronous systems', 'measurements', 'system model', 'synchronous systems', 'timed model', 'communication by time'], ['real-time', 'scheduling theory', 'uniprocessor scheduling', 'multiprocessor scheduling'], ['fault tolerance', 'checkpointing', 'imprecise computation', 'target tracking', 'distributed real-time systems', 'beam forming'], ['networks', 'scheduling', 'NP-completeness', 'approximation algorithm'], ['real-time scheduling', 'defragmentation', 'read barrier', 'utilization'], ['consistent global states', 'distributed debugging', 'global predicate detection', 'real-time monitoring'], ['approximation algorithm', 'job scheduling', 'multiprocessor processing', 'polynomial time approximation scheme'], ['admission control', 'plus algebra', 'regular ordering balanced sequences', 'schur convexity'], ['asynchronous distributed computation', 'contractibility problem', 'task-solvability', 'wait-free computation'], ['fault-tolerance', 'transactions', 'replication', 'end-to-end reliability', 'exactly-once semantics'], ['fault tolerance', 'real-time system', 'dynamic scheduling', 'resource reclaiming', 'run-time anomaly', 'safety critical application'], ['scheduling', 'approximation algorithm', 'on-line algorithm', 'LP relaxation'], ['distributed processing', 'fault-tolerance', 'reliability', 'fail-silence', 'replicated processing'], ['discrete event simulation', 'speedup', 'rollback overhead', 'straggler', 'antimessage', 'breadth-first rollback', 'causal relationship recovery', 'incremental state saving', 'optimistic protocol', 'parallel discrete event simulations', 'rollback processing', 'simulation objects', 'spatially explicit simulations'], ['fault tolerance', 'scheduling', 'real-time systems', 'temporal consistency', 'replication protocols'], ['real-time scheduling', 'earliest-deadline first', 'fault-tolerant schedules', 'fault recovery'], ['functional programming', 'programming theory', 'exception handling', 'high level languages', 'functional languages', 'commutativity properties', 'implementation restriction', 'input object', 'programmer', 'referential transparency', 'resume', 'terminate'], ['distributed systems', 'masking and nonmasking fault-tolerance', 'component based design', 'correctors', 'detectors', 'stepwise design formal methods'], ['sather', 'general control structures', 'iteration abstraction'], ['data structures', 'persistent data structures', 'queue', 'stack', 'catenation', 'double-ended queue', 'purely functional data structures', 'purely functional queues'], ['scheduling', 'real-time systems and applications', 'imprecise computation', 'error', 'end-to-end timing constraints'], ['or-parallel execution models'], ['recursive programming', 'separation of concerns', 'visitor pattern'], ['scheduling', 'fairness', 'dining philosophers problem', 'interval graphs'], ['distributed systems', 'on-the-fly global predicate detection'], ['approximation algorithm', 'combinatorial optimization', 'scheduling theory', 'approximation scheme', 'worst-case ratio'], ['self-stabilization', 'clock synchronization', 'byzantine failures'], ['scheduling', 'resource augmentation', 'multi-level feedback scheduling'], ['concurrency', 'multiprogramming', 'communicating processes', 'distributed implementation', 'object-based programming', 'semaphore'], ['exception handling', 'software fault tolerance', 'conversations', 'atomic actions', 'ada 95', 'recovery blocks'], ['scheduling', 'delay bounds', 'earliest deadline first', 'packet routing', 'weighted fair queueing'], ['GI/G/1 queue', 'fluid approximation', 'functional strong law of large numbers', 'processor-sharing discipline'], ['storage management', 'transaction processing', 'checkpointing', 'system recovery', 'updates', 'database management systems', 'high-level recovery manager', 'incremental database recovery', 'memory database systems', 'nonvolatile RAM', 'online transaction processing systems', 'page-based incremental restart algorithm'], ['simulation', 'Generalized Semi-Markov Processes', 'gradient estimation'], ['online algorithms', 'competitive analysis', 'metrical task systems', 'paging problem'], ['distributed systems', 'fault tolerance', 'group communication', 'piecewise determinism', 'semi-active replication'], ['queues', 'resource sharing', 'service systems', 'multiserver queues', 'Service Systems with Express Lines', 'service-system design'], ['fault tolerance', 'mapping', 'architecture', 'error detection', 'multicomputers', 'fault-tolerant algorithms'], ['linearization', 'wait-free synchronization'], ['M/G/infty process', 'finite buffer queue', 'fluid flow queue', 'long-range dependency', 'long-tailed traffic models', 'network multiplexer', 'subexponential distributions'], ['queuing networks', 'randomized algorithms'], ['online computation', 'hierarchical cooperative caching'], ['distributed systems', 'protocols', 'performance evaluation', 'fault tolerance', 'checkpointing', 'rollback-recovery', 'causal dependency', 'timestamp management', 'global snapshot'], ['proof-carrying code', 'typed assembly language'], ['linear programming', 'integer programming', 'derandomization', 'linear codes', 'set cover', 'resource sharing', 'probabilistic algorithms', 'distributed networks', 'file assignment'], ['fault-tolerance', 'concurrency', 'synchronization', 'distributed algorithms', 'lower bounds'], ['distributed real-time systems', 'adaptive resource allocation', 'asynchronous real-time distributed systems', 'proactive resource allocation', 'best-effort resource allocation', 'best-effort real-time scheduling', 'benefit functions', 'switched real-time Ethernet'], ['admission control', 'notification', 'online scheduling', 'firm deadlines'], ['fault-tolerance', 'link failures', 'cellular arrays', 'static and dynamic defects', 'syntactical pattern recognition'], ['security', 'graph algorithms', 'crowns', 'partial orders', 'distributed networks', 'hierarchically decomposed databases', 'multilevel databases'], ['message passing', 'communication', 'object-oriented hardware modelling'], ['garbage collection', 'generational and copy collection', 'object behavior', 'write barrier'], ['shared memory systems', 'distributed systems', 'atomic read/write registers', 'combinatorial topology', 'consensus', 'renaming', 'set consensus', 'wait-free solvable tasks'], ['balanced sequences', 'multimodularity', 'optimal control', 'stochastic event graphs'], ['programming languages', 'memory management', 'garbage collection', 'generational garbage collection'], ['distributed systems', 'mutual exclusion', 'fault-tolerant computing', 'quorum consensus', 'replicated database systems'], ['abandonments', 'balking', 'Birth-and-Death Processes', 'Communicating Anticipated Delays', 'reneging', 'retrials', 'service systems', 'Telephone Call Centers'], ['fairness', 'program verification', 'distributed consensus', 'modeling of asynchronous systems'], ['performance analysis', 'concurrent system', 'queuing system', 'reader/writer', 'lock queue'], ['load balancing', 'on-line algorithms', 'hierarchical servers', 'resource procurement', 'temporary jobs'], ['shared memory systems', 'release consistency', 'index termsmultiprocessors', 'data-race-free-0', 'data-race-free-1', 'formalization', 'hazards and raceconditions', 'sequential consistency', 'shared-memory models', 'weak ordering'], ['distributed algorithms', 'distributed languages', 'multiparty interaction', 'committee coordination', 'first-order interaction', 'interaction scheduling', 'rendezvous'], ['semantics', 'language design', 'generic functions', 'binary methods', 'multimethods', 'multiple dispatch', 'single dispatch', 'tuple', 'typing'], ['time warp', 'checkpointing', 'rollback-recovery', 'performance optimization', 'optimistic synchronization', 'parallel discrete-event simulation', 'cost models'], ['fault-tolerance', 'distributed shared memory', 'portability', 'checkpointing'], ['distributed computing', 'distributed systems', 'modularity', 'distributed processing', 'fault tolerance', 'fault tolerant computing', 'distributed environment', 'replication', 'object-oriented approach', 'atomic actions', 'index termsobject-oriented methods', 'atomic transactions', 'distributed programming systems', 'fault-tolerantobject systems', 'migration', 'object-oriented systems', 'persistent objects'], ['discrete event simulation', 'time warp', 'performance measurements', 'parallel', 'distributed', 'time warp simulation', 'PCS networks', 'flexible analysis tools', 'mobile communication', 'mobile communications', 'personal communication service', 'simulation platforms', 'telecommunication computing'], ['PVM', 'time warp', 'petri nets', 'RS6000 cluster', 'forecast models', 'optimism control'], ['asymptotic relationship', 'loss probability', 'maximum variance asymptotic', 'queue length distribution'], ['data structures', 'shortest paths', 'priority queues'], ['functional programming', 'data structures', 'memoization', 'double-ended queue deque', 'persistent data structures', 'queue', 'stack', 'stack-ended queue steque'], ['self-stabilization', 'communication networks', 'crash failures', 'end-to-end protocols', 'dynamic networks'], ['real-time systems', 'imprecise computation', 'reward maximization', 'periodic task scheduling', 'deadline scheduling'], ['distributed systems', 'design synthesis', 'embedded systems', 'soft real-time', 'statistical performance'], ['distributed computing', 'performance evaluation', 'message passing', 'parallel programming', 'compiler', 'synchronisation', 'graph theory', 'exponential distribution', 'random variables', 'distributed networks', 'index termsprogram compilers', 'asynchronous network', 'bottleneck processor', 'computational steps', 'marked graphs', 'message transmissiondelays', 'performance measure', 'probabilitydistributions', 'processing times', 'randomprocessing times', 'synchronized programs', 'synchronizer', 'synchronous network', 'transmission delays'], ['coverage', 'anomaly', 'anomaly detection', 'dependability'], ['fault tolerance', 'quality of service', 'distributed algorithm', 'probabilistic analysis', 'failure detectors'], ['clock synchronization', 'mobile adversary', 'proactive systems'], ['debugging', 'distributed memory systems', 'testing', 'distributed algorithms', 'program debugging', 'UNIX', 'distributed programs', 'message complexity', 'distributed debugger', 'index termscommunication complexity', 'sun workstations', 'global predicates', 'programtesting', 'weak unstable predicates', 'weakconjunctive predicates'], ['scheduling', 'fairness', 'game theory', 'nash equilibrium', 'TCP', 'congestion control', 'GPS', 'DWS', 'RIS', 'generalized processor sharing', 'stackelberg equilibrium'], ['fault detection', 'replicated data', 'quorum systems', 'byzantine fault tolerance'], ['distributed algorithms', 'lower bounds', 'fault', 'tolerence', 'work complexity'], ['caching', 'randomized', 'online', 'application-controlled', 'competitive'], ['real-time systems', 'resource constraints', 'dynamic scheduling', 'multiprocessor', 'parallelizable tasks'], ['self-stabilization', 'parallel programming', 'formal logic', 'adaptive systems', 'programming theory', 'adaptivity', 'token ring networks', 'adaptive distributed programs', 'adaptive sequential programs', 'composition operators', 'constituent programs'], ['load balancing', 'load sharing', 'fairness', 'task assignment', 'job scheduling', 'clusters', 'contrary behavior', 'distributed servers', 'heavy-tailed workloads', 'high variance', 'supercomputing'], ['processor allocation', 'resource allocation', 'load balancing', 'task assignment', 'task allocation', 'majorization', 'performance of parallel systems'], ['quorum systems', 'byzantine fault tolerance', 'atomic variable semantics', 'distributed data services'], ['matching', 'online algorithms', 'competitive analysis'], ['concurrency', 'concurrent object-oriented programming', 'ada 95', 'inheritance anomaly'], ['fault tolerance', 'rollback recovery', 'staggered checkpoints', 'consistent recovery line', 'stable storage contention'], ['reconfigurable architectures', 'lower bound', 'fault tolerant computing', 'reliability', 'time complexity', 'reconfigurability', 'systolic arrays', 'application graph', 'bounded-degree graphs', 'dynamic graphs', 'fault-tolerant redundant structures', 'finitely reconfigurable', 'locally reconfigurable', 'reliable arrays', 'wavefront arrays'], ['logic programming', 'parallel programming', 'parallel languages', 'GHC', 'parlog', 'clean semantics', 'concurrent logic programming languages', 'control facilities', 'execution conditions', 'language translation', 'lingua franca', 'nonflat guards'], ['coherence protocol', 'event ordering', 'hardware DSM systems', 'memory consistency models', 'software DSM systems'], ['parallel and distributed simulation', 'discrete-event simulation', 'lookahead', 'conservative algorithms', 'parallel simulation languages', 'algorithmic efficiency'], ['scheduling', 'approximation algorithm', 'on-line algorithm', 'randomized rounding', 'linear programming relaxation'], ['fault tolerant computing', 'software maintenance', 'software reliability', 'redundancy', 'process computer control', 'system reliability', 'burst hardware failures', 'cold standby', 'computer integrated manufacturing', 'hierarchically structured process-control programs', 'hot standby', 'long-lived unmaintainable systems', 'process-control programs', 'simulated chemical batch reactor system', 'standby redundancy design space', 'warm standby'], ['atomic register', 'linearizability', 'wait-free synchronization'], ['performance modeling', 'parallel processing', 'synchronization', 'lock-free', 'nonblocking'], ['shared memory', 'DSM', 'data races', 'on-the-fly'], ['checkpointing', 'distributed shared memory system', 'fault tolerant system', 'message logging', 'rollback-recovery'], ['concurrency', 'continuations', 'threads', 'control delimiters', 'control operators'], ['communication protocols', 'fault-tolerant systems', 'distributed real-time systems']]
# t = []
# for te in test:
#     print(te)
#     # t.append(te)
#     t = t + te
#
# print(t)

# x = [1,2,3]
# print(x[:2])