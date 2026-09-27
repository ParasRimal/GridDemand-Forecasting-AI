import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getModelingFindings } from "../api/endpoints";

export default function About() {
  const [content, setContent] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelingFindings()
      .then((data) => setContent(data.content))
      .catch((err) => setError(err.response?.data?.error || err.message));
  }, []);

  if (error) {
    return (
      <div className="max-w-2xl mx-auto mt-12 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
        Failed to load findings: {error}
      </div>
    );
  }
  if (!content) return <p className="text-center mt-12 text-gray-500">Loading...</p>;

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <article
        className="prose prose-gray dark:prose-invert max-w-none
          prose-h1:text-3xl prose-h1:font-bold
          prose-h2:text-xl prose-h2:font-semibold prose-h2:mt-8
          prose-table:text-sm
          bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-8"
      >
        <ReactMarkdown>{content}</ReactMarkdown>
      </article>
    </div>
  );
}
