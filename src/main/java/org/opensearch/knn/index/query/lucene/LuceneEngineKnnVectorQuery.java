/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.lucene;

import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TopKnnCollector;
import org.apache.lucene.search.Weight;
import org.opensearch.knn.index.query.lucenelib.ExpandNestedDocsQuery;
import org.opensearch.knn.profile.KNNProfileUtil;
import org.opensearch.search.profile.query.QueryProfiler;
import org.opensearch.knn.index.query.common.QueryUtils;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * LuceneEngineKnnVectorQuery is a wrapper around a vector queries for the Lucene engine.
 * This enables us to defer rewrites until weight creation to optimize repeated execution
 * of Lucene based k-NN queries.
 */
@AllArgsConstructor
@EqualsAndHashCode(callSuper = false)
@Log4j2
public class LuceneEngineKnnVectorQuery extends Query {
    @Getter
    private final Query luceneQuery;
    private final int luceneK; // Number of results requested from Lucene engine (may be > k for better recall)
    private final int k; // Final number of results to return to user

    /*
      Prevents repeated rewrites of the query for the Lucene engine.
    */
    @Override
    public Query rewrite(IndexSearcher indexSearcher) {
        return this;
    }

    /*
       Rewrites the query just before weight creation.
     */
    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        QueryProfiler profiler = KNNProfileUtil.getProfiler(searcher);
        if (profiler != null) {
            profiler.getQueryBreakdown(luceneQuery);
        }
        Query rewrittenQuery = luceneQuery.rewrite(searcher);
        Query docAndScoreQuery = reduceToTopK(rewrittenQuery, searcher, scoreMode, boost);
        final Weight weight = docAndScoreQuery.createWeight(searcher, scoreMode, boost);
        if (profiler != null) {
            profiler.pollLastElement();
        }
        return weight;
    }

    private Query reduceToTopK(Query query, IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {

        // Skip reducing to top-k in two cases:
        // 1. When luceneK equals k (no reduction needed)
        // 2. When query is ExpandNestedDocsQuery (reducing would exclude required child documents)
        if (luceneK == k || query instanceof ExpandNestedDocsQuery) {
            return query;
        }

        Weight weight = query.createWeight(searcher, scoreMode, boost);

        // Collect results from all leaves in parallel
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();
        List<Map<Integer, Float>> leafResults = QueryUtils.getInstance().doSearch(searcher, leaves, weight);

        // Add all results to collector sequentially, converting leaf-relative to global doc IDs
        TopKnnCollector collector = new TopKnnCollector(k, Integer.MAX_VALUE);
        for (int i = 0; i < leafResults.size(); i++) {
            LeafReaderContext context = leaves.get(i);
            Map<Integer, Float> leafResult = leafResults.get(i);
            for (Map.Entry<Integer, Float> entry : leafResult.entrySet()) {
                collector.collect(entry.getKey() + context.docBase, entry.getValue());
            }
        }

        return QueryUtils.getInstance().createDocAndScoreQuery(searcher.getIndexReader(), collector.topDocs());
    }

    @Override
    public String toString(String s) {
        return "LuceneEngineKnnVectorQuery[luceneK=" + luceneK + ", k=" + k + ", query=" + luceneQuery.toString() + "]";
    }

    @Override
    public void visit(QueryVisitor queryVisitor) {
        queryVisitor.visitLeaf(this);
    }
}
