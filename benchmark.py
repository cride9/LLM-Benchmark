import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    # Simple tqdm replacement if not installed
    def tqdm(iterable, **kwargs):
        print(f"Processing {len(iterable)} items...")
        return iterable

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ollama_benchmark")

class ContextBenchmark:
    """
    A benchmark tool for testing LLM context length capabilities with a focus on memory retention.
    """
    
    def __init__(self, model_name: str, output_dir: str = "results", test_mode: bool = False):
        """
        Initialize the benchmark with the specified model.
        
        Args:
            model_name: Name of the Ollama model to benchmark
            output_dir: Directory to save benchmark results
            test_mode: If True, run in test mode without requiring Ollama
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.test_mode = test_mode
        self.results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Test connection to Ollama if not in test mode
        if not test_mode:
            if not OLLAMA_AVAILABLE:
                logger.error("Ollama Python package is not installed. Please install it with 'pip install ollama'")
                raise ImportError("Ollama Python package is not installed")
            
            try:
                self._test_ollama_connection()
                logger.info(f"Successfully connected to Ollama with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                raise
        else:
            logger.info("Running in test mode, Ollama connection check skipped")
    
    def _test_ollama_connection(self):
        """Test connection to Ollama and verify the model exists."""
        try:
            models = ollama.list()
            # Fix: Use 'model' key instead of 'name'
            model_names = [model['model'] for model in models['models']]
            
            if self.model_name not in model_names:
                available_models = ", ".join(model_names)
                raise ValueError(f"Model '{self.model_name}' not found. Available models: {available_models}")
        except Exception as e:
            if "connection refused" in str(e).lower():
                raise ConnectionError("Could not connect to Ollama. Is the Ollama server running?") from e
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a string to be used as a filename.
        
        Args:
            filename: The string to sanitize
            
        Returns:
            Sanitized string safe for use as a filename
        """
        # Replace characters that are problematic in filenames
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        return sanitized
    
    def _generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, float]:
        """
        Generate a response from the model and measure the time taken.
        
        Args:
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            
        Returns:
            Tuple of (response_text, time_taken_seconds)
        """
        if self.test_mode:
            # In test mode, simulate a response
            time.sleep(0.5)  # Simulate processing time
            return f"Test response for: {prompt[:50]}...", 0.5
        
        start_time = time.time()
        
        try:
            if system_prompt:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                )
            else:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            end_time = time.time()
            time_taken = end_time - start_time
            
            return response['message']['content'], time_taken
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"ERROR: {str(e)}", time.time() - start_time
    
    def run_basic_retrieval_test(self, context_lengths: List[int] = [1000, 5000, 10000, 20000]):
        """
        Run the basic retrieval test at different context lengths.
        
        This test places key facts at different positions in the context and tests if the model
        can retrieve them accurately.
        
        Args:
            context_lengths: List of context lengths to test
        """
        logger.info(f"Running Basic Retrieval Test with context lengths: {context_lengths}")
        
        results = {
            "test_name": "basic_retrieval",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results_by_length": {}
        }
        
        for context_length in tqdm(context_lengths, desc="Testing context lengths"):
            # Generate test data
            test_data = self._generate_retrieval_test_data(context_length)
            
            position_results = {}
            for position, test_case in test_data.items():
                context, question, expected_answer = test_case
                
                # Run the test
                prompt = f"{context}\n\nQuestion: {question}"
                response, time_taken = self._generate_response(prompt)
                
                # Evaluate the response
                correct, match_details = self._evaluate_retrieval_response(response, expected_answer)
                
                position_results[position] = {
                    "correct": correct,
                    "time_taken": time_taken,
                    "expected": expected_answer,
                    "response": response,
                    "match_details": match_details
                }
                
                # Log the result
                logger.info(f"Context length: {context_length}, Position: {position}, Correct: {correct}, Time: {time_taken:.2f}s")
                if not correct:
                    logger.info(f"Match details: {match_details}")
            
            # Calculate accuracy for this context length
            accuracy = sum(1 for r in position_results.values() if r["correct"]) / len(position_results)
            
            results["results_by_length"][context_length] = {
                "accuracy": accuracy,
                "position_results": position_results
            }
        
        # Save results
        self.results["basic_retrieval"] = results
        self._save_results("basic_retrieval")
        
        return results
    
    def run_memory_association_test(self, context_lengths: List[int] = [1000, 5000, 10000, 20000]):
        """
        Run the memory association test at different context lengths.
        
        This test inserts related facts throughout the context and tests if the model
        can connect them correctly.
        
        Args:
            context_lengths: List of context lengths to test
        """
        logger.info(f"Running Memory Association Test with context lengths: {context_lengths}")
        
        results = {
            "test_name": "memory_association",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results_by_length": {}
        }
        
        for context_length in tqdm(context_lengths, desc="Testing context lengths"):
            # Generate test data
            test_data = self._generate_association_test_data(context_length)
            
            test_results = {}
            for test_id, test_case in test_data.items():
                context, question, expected_answer = test_case
                
                # Run the test
                prompt = f"{context}\n\nQuestion: {question}"
                response, time_taken = self._generate_response(prompt)
                
                # Evaluate the response
                correct, match_details = self._evaluate_association_response(response, expected_answer)
                
                test_results[test_id] = {
                    "correct": correct,
                    "time_taken": time_taken,
                    "expected": expected_answer,
                    "response": response,
                    "match_details": match_details
                }
                
                # Log the result
                logger.info(f"Context length: {context_length}, Test ID: {test_id}, Correct: {correct}, Time: {time_taken:.2f}s")
                if not correct:
                    logger.info(f"Match details: {match_details}")
            
            # Calculate accuracy for this context length
            accuracy = sum(1 for r in test_results.values() if r["correct"]) / len(test_results)
            
            results["results_by_length"][context_length] = {
                "accuracy": accuracy,
                "test_results": test_results
            }
        
        # Save results
        self.results["memory_association"] = results
        self._save_results("memory_association")
        
        return results
    
    def run_multi_document_test(self, context_lengths: List[int] = [1000, 5000, 10000, 20000]):
        """
        Run the multi-document summarization test at different context lengths.
        
        This test creates a context with multiple "documents" and tests if the model
        can create a coherent summary that includes key points from all documents.
        
        Args:
            context_lengths: List of context lengths to test
        """
        logger.info(f"Running Multi-Document Test with context lengths: {context_lengths}")
        
        results = {
            "test_name": "multi_document",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results_by_length": {}
        }
        
        for context_length in tqdm(context_lengths, desc="Testing context lengths"):
            # Generate test data
            test_data = self._generate_multi_document_test_data(context_length)
            
            test_results = {}
            for test_id, test_case in test_data.items():
                context, question, key_points = test_case
                
                # Run the test
                prompt = f"{context}\n\nTask: {question}"
                response, time_taken = self._generate_response(prompt)
                
                # Evaluate the response
                score, details = self._evaluate_multi_document_response(response, key_points)
                
                test_results[test_id] = {
                    "score": score,
                    "time_taken": time_taken,
                    "key_points": key_points,
                    "response": response,
                    "evaluation_details": details
                }
                
                # Log the result
                logger.info(f"Context length: {context_length}, Test ID: {test_id}, Score: {score:.2f}, Time: {time_taken:.2f}s")
            
            # Calculate average score for this context length
            avg_score = sum(r["score"] for r in test_results.values()) / len(test_results)
            
            results["results_by_length"][context_length] = {
                "average_score": avg_score,
                "test_results": test_results
            }
        
        # Save results
        self.results["multi_document"] = results
        self._save_results("multi_document")
        
        return results
    
    def run_topic_switching_test(self, context_lengths: List[int] = [1000, 5000, 10000, 20000]):
        """
        Run the topic switching test at different context lengths.
        
        This test interleaves information about multiple topics and tests if the model
        can maintain topic coherence when asked to focus on one topic.
        
        Args:
            context_lengths: List of context lengths to test
        """
        logger.info(f"Running Topic Switching Test with context lengths: {context_lengths}")
        
        results = {
            "test_name": "topic_switching",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results_by_length": {}
        }
        
        for context_length in tqdm(context_lengths, desc="Testing context lengths"):
            # Generate test data
            test_data = self._generate_topic_switching_test_data(context_length)
            
            test_results = {}
            for test_id, test_case in test_data.items():
                context, question, relevant_facts, irrelevant_facts = test_case
                
                # Run the test
                prompt = f"{context}\n\nQuestion: {question}"
                response, time_taken = self._generate_response(prompt)
                
                # Evaluate the response
                score, details = self._evaluate_topic_switching_response(response, relevant_facts, irrelevant_facts)
                
                test_results[test_id] = {
                    "score": score,
                    "time_taken": time_taken,
                    "relevant_facts": relevant_facts,
                    "irrelevant_facts": irrelevant_facts,
                    "response": response,
                    "evaluation_details": details
                }
                
                # Log the result
                logger.info(f"Context length: {context_length}, Test ID: {test_id}, Score: {score:.2f}, Time: {time_taken:.2f}s")
            
            # Calculate average score for this context length
            avg_score = sum(r["score"] for r in test_results.values()) / len(test_results)
            
            results["results_by_length"][context_length] = {
                "average_score": avg_score,
                "test_results": test_results
            }
        
        # Save results
        self.results["topic_switching"] = results
        self._save_results("topic_switching")
        
        return results
    
    def run_all_tests(self, context_lengths: List[int] = [1000, 5000, 10000, 20000]):
        """
        Run all benchmark tests with the specified context lengths.
        
        Args:
            context_lengths: List of context lengths to test
        """
        logger.info(f"Running all benchmark tests with model: {self.model_name}")
        logger.info(f"Context lengths to test: {context_lengths}")
        
        # Run all tests
        self.run_basic_retrieval_test(context_lengths)
        self.run_memory_association_test(context_lengths)
        self.run_multi_document_test(context_lengths)
        self.run_topic_switching_test(context_lengths)
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("All benchmark tests completed successfully")
    
    def _generate_summary_report(self):
        """Generate a summary report of all test results."""
        summary = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Summarize each test
        for test_name, results in self.results.items():
            test_summary = {
                "context_lengths": list(results["results_by_length"].keys())
            }
            
            # Extract accuracy or score for each context length
            if test_name in ["basic_retrieval", "memory_association"]:
                test_summary["accuracy_by_length"] = {
                    str(length): data["accuracy"] 
                    for length, data in results["results_by_length"].items()
                }
            else:
                test_summary["score_by_length"] = {
                    str(length): data["average_score"] 
                    for length, data in results["results_by_length"].items()
                }
            
            summary["tests"][test_name] = test_summary
        
        # Save summary
        # Fix: Sanitize model name for use in filename
        sanitized_model_name = self._sanitize_filename(self.model_name)
        summary_path = os.path.join(self.output_dir, f"{sanitized_model_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary
    
    def _save_results(self, test_name: str):
        """Save test results to a JSON file."""
        # Fix: Sanitize model name for use in filename
        sanitized_model_name = self._sanitize_filename(self.model_name)
        results_path = os.path.join(self.output_dir, f"{sanitized_model_name}_{test_name}.json")
        with open(results_path, 'w') as f:
            json.dump(self.results[test_name], f, indent=2)
        
        logger.info(f"{test_name} results saved to {results_path}")
    
    # Test data generation methods
    def _generate_retrieval_test_data(self, context_length: int) -> Dict[str, Tuple[str, str, str]]:
        """
        Generate test data for the basic retrieval test.
        
        Args:
            context_length: Target length of the context
            
        Returns:
            Dictionary mapping position names to (context, question, expected_answer) tuples
        """
        # Get filler text to pad the context
        filler_text = self._get_filler_text(context_length)
        filler_paragraphs = filler_text.split('\n\n')
        
        # Create unique facts to insert at different positions
        facts = {
            "beginning": "The rare purple mongoose (Herpestes purpureus) was discovered in 2018 by Dr. Emily Chen in the highlands of Madagascar.",
            "early": "The ancient city of Avaloria was founded in 327 BCE and was known for its advanced water filtration system using volcanic rocks.",
            "middle": "The Zephyr Protocol, developed by physicist Dr. Marcus Wei in 2021, demonstrated quantum entanglement across a distance of 15 kilometers.",
            "late": "The Crimson Nebula contains an unusual concentration of beryllium-7, making it a prime target for the James Webb Space Telescope in 2026.",
            "end": "The world's largest collection of antique music boxes is housed in the Melodia Museum in Vienna, with 4,827 unique specimens dating from 1796 to 1935."
        }
        
        # Questions corresponding to each fact
        questions = {
            "beginning": "When and by whom was the rare purple mongoose discovered?",
            "early": "When was the ancient city of Avaloria founded and what was it known for?",
            "middle": "What is the Zephyr Protocol and who developed it?",
            "late": "What makes the Crimson Nebula a prime target for the James Webb Space Telescope?",
            "end": "Where is the world's largest collection of antique music boxes housed and how many specimens does it contain?"
        }
        
        # Expected answers
        expected_answers = {
            "beginning": "The rare purple mongoose was discovered in 2018 by Dr. Emily Chen in the highlands of Madagascar.",
            "early": "The ancient city of Avaloria was founded in 327 BCE and was known for its advanced water filtration system using volcanic rocks.",
            "middle": "The Zephyr Protocol was developed by physicist Dr. Marcus Wei in 2021 and demonstrated quantum entanglement across a distance of 15 kilometers.",
            "late": "The Crimson Nebula contains an unusual concentration of beryllium-7, making it a prime target for the James Webb Space Telescope in 2026.",
            "end": "The world's largest collection of antique music boxes is housed in the Melodia Museum in Vienna, with 4,827 unique specimens dating from 1796 to 1935."
        }
        
        # Calculate positions to insert facts
        total_paragraphs = len(filler_paragraphs)
        positions = {
            "beginning": 0,
            "early": total_paragraphs // 10,
            "middle": total_paragraphs // 2,
            "late": total_paragraphs * 8 // 10,
            "end": total_paragraphs - 1
        }
        
        # Create test cases for each position
        test_data = {}
        for position_name, position_idx in positions.items():
            # Copy filler paragraphs
            context_paragraphs = filler_paragraphs.copy()
            
            # Insert the fact at the specified position
            context_paragraphs[position_idx] = facts[position_name]
            
            # Join paragraphs to create the context
            context = '\n\n'.join(context_paragraphs)
            
            # Trim or pad to match target length
            context = self._adjust_text_length(context, context_length)
            
            test_data[position_name] = (context, questions[position_name], expected_answers[position_name])
        
        return test_data
    
    def _generate_association_test_data(self, context_length: int) -> Dict[str, Tuple[str, str, str]]:
        """
        Generate test data for the memory association test.
        
        Args:
            context_length: Target length of the context
            
        Returns:
            Dictionary mapping test IDs to (context, question, expected_answer) tuples
        """
        # Get filler text to pad the context
        filler_text = self._get_filler_text(context_length)
        filler_paragraphs = filler_text.split('\n\n')
        
        # Create related facts to insert at different positions
        test_cases = [
            {
                "facts": [
                    "Company Alpha was founded in 1985 by Dr. James Wilson.",
                    "Dr. James Wilson previously worked at MIT's Artificial Intelligence Lab.",
                    "Company Alpha's first product was the DataMind 1000, released in 1987.",
                    "The DataMind 1000 used a revolutionary neural network algorithm developed at MIT."
                ],
                "question": "What is the connection between Company Alpha's first product and its founder's background?",
                "expected_answer": "Company Alpha's first product, the DataMind 1000, used a revolutionary neural network algorithm developed at MIT, where the founder Dr. James Wilson previously worked in the Artificial Intelligence Lab."
            },
            {
                "facts": [
                    "The ancient city of Lyranth was built near the confluence of the Azura and Crimson rivers.",
                    "Archaeologists discovered that Lyranth was abandoned around 1200 CE due to a severe drought.",
                    "Climate studies of the region show that the Azura river completely dried up between 1198-1205 CE.",
                    "The Crimson river's water level dropped by 80% during the same period."
                ],
                "question": "What evidence supports the theory about why Lyranth was abandoned?",
                "expected_answer": "Lyranth was abandoned around 1200 CE due to a severe drought, which is supported by climate studies showing that the Azura river completely dried up between 1198-1205 CE and the Crimson river's water level dropped by 80% during the same period. The city was built near the confluence of these two rivers."
            },
            {
                "facts": [
                    "The Monarch butterfly (Danaus plexippus) migrates up to 3,000 miles each year.",
                    "Monarch caterpillars exclusively feed on milkweed plants.",
                    "Milkweed contains cardiac glycosides, which are toxic to most animals.",
                    "Monarchs have evolved to store these toxins in their bodies, making them poisonous to predators."
                ],
                "question": "Explain the relationship between Monarch butterflies, milkweed, and predators.",
                "expected_answer": "Monarch caterpillars exclusively feed on milkweed plants, which contain cardiac glycosides that are toxic to most animals. Monarchs have evolved to store these toxins in their bodies, making them poisonous to predators."
            }
        ]
        
        # Calculate positions to insert facts
        total_paragraphs = len(filler_paragraphs)
        
        test_data = {}
        for i, test_case in enumerate(test_cases):
            # Copy filler paragraphs
            context_paragraphs = filler_paragraphs.copy()
            
            # Calculate positions to insert facts (spread throughout the context)
            num_facts = len(test_case["facts"])
            step = total_paragraphs // (num_facts + 1)
            positions = [step * (j + 1) for j in range(num_facts)]
            
            # Insert facts at calculated positions
            for j, position in enumerate(positions):
                if position < len(context_paragraphs):
                    context_paragraphs[position] = test_case["facts"][j]
            
            # Join paragraphs to create the context
            context = '\n\n'.join(context_paragraphs)
            
            # Trim or pad to match target length
            context = self._adjust_text_length(context, context_length)
            
            test_data[f"association_{i+1}"] = (context, test_case["question"], test_case["expected_answer"])
        
        return test_data
    
    def _generate_multi_document_test_data(self, context_length: int) -> Dict[str, Tuple[str, str, List[str]]]:
        """
        Generate test data for the multi-document summarization test.
        
        Args:
            context_length: Target length of the context
            
        Returns:
            Dictionary mapping test IDs to (context, question, key_points) tuples
        """
        # Define document topics
        topics = [
            "Renewable Energy",
            "Artificial Intelligence",
            "Space Exploration"
        ]
        
        # Create documents for each topic with key points
        documents = {
            "Renewable Energy": {
                "content": self._get_topic_text("renewable energy", 2000),
                "key_points": [
                    "Solar power capacity has increased tenfold in the last decade",
                    "Wind energy is now cost-competitive with fossil fuels in many regions",
                    "Energy storage remains a significant challenge for renewable adoption",
                    "Government incentives have been crucial for renewable energy growth"
                ]
            },
            "Artificial Intelligence": {
                "content": self._get_topic_text("artificial intelligence", 2000),
                "key_points": [
                    "Deep learning has revolutionized computer vision and natural language processing",
                    "Ethical concerns include bias in AI systems and job displacement",
                    "Large language models can generate human-like text but struggle with factual accuracy",
                    "AI regulation is being developed in various countries"
                ]
            },
            "Space Exploration": {
                "content": self._get_topic_text("space exploration", 2000),
                "key_points": [
                    "Private companies have significantly reduced the cost of space launches",
                    "Mars missions are planned for the late 2020s and early 2030s",
                    "The search for extraterrestrial life focuses on ocean moons like Europa",
                    "Space telescopes have revolutionized our understanding of the universe"
                ]
            }
        }
        
        # Get filler text to pad the context if needed
        filler_text = self._get_filler_text(context_length // 4)
        
        test_data = {}
        
        # Create a test case that combines all documents
        all_docs_content = ""
        all_key_points = []
        
        for topic in topics:
            all_docs_content += f"\n\n## Document: {topic}\n\n"
            all_docs_content += documents[topic]["content"]
            all_key_points.extend(documents[topic]["key_points"])
        
        # Add filler if needed to reach target length
        if len(all_docs_content) < context_length:
            all_docs_content += f"\n\n## Additional Information\n\n{filler_text}"
        
        # Trim to match target length
        context = self._adjust_text_length(all_docs_content, context_length)
        
        question = f"Please summarize the key points from all the documents about {', '.join(topics)}."
        
        test_data["multi_doc_all"] = (context, question, all_key_points)
        
        # Create test cases for pairs of documents
        for i in range(len(topics)):
            for j in range(i+1, len(topics)):
                topic1, topic2 = topics[i], topics[j]
                
                combined_content = f"\n\n## Document: {topic1}\n\n"
                combined_content += documents[topic1]["content"]
                combined_content += f"\n\n## Document: {topic2}\n\n"
                combined_content += documents[topic2]["content"]
                
                # Add filler if needed to reach target length
                if len(combined_content) < context_length:
                    combined_content += f"\n\n## Additional Information\n\n{filler_text}"
                
                # Trim to match target length
                context = self._adjust_text_length(combined_content, context_length)
                
                combined_key_points = documents[topic1]["key_points"] + documents[topic2]["key_points"]
                
                question = f"Please summarize the key points from the documents about {topic1} and {topic2}."
                
                test_data[f"multi_doc_{topic1}_{topic2}"] = (context, question, combined_key_points)
        
        return test_data
    
    def _generate_topic_switching_test_data(self, context_length: int) -> Dict[str, Tuple[str, str, List[str], List[str]]]:
        """
        Generate test data for the topic switching test.
        
        Args:
            context_length: Target length of the context
            
        Returns:
            Dictionary mapping test IDs to (context, question, relevant_facts, irrelevant_facts) tuples
        """
        # Define pairs of similar but distinct topics
        topic_pairs = [
            ("quantum physics", "astronomy"),
            ("machine learning", "neuroscience"),
            ("ancient rome", "ancient greece")
        ]
        
        test_data = {}
        
        for pair_idx, (topic1, topic2) in enumerate(topic_pairs):
            # Generate facts for each topic
            topic1_facts = self._generate_facts_for_topic(topic1, 10)
            topic2_facts = self._generate_facts_for_topic(topic2, 10)
            
            # Interleave the facts
            interleaved_facts = []
            for i in range(max(len(topic1_facts), len(topic2_facts))):
                if i < len(topic1_facts):
                    interleaved_facts.append(topic1_facts[i])
                if i < len(topic2_facts):
                    interleaved_facts.append(topic2_facts[i])
            
            # Get filler text to pad the context
            filler_paragraphs = self._get_filler_text(context_length).split('\n\n')
            
            # Calculate positions to insert facts
            total_paragraphs = len(filler_paragraphs)
            step = total_paragraphs // (len(interleaved_facts) + 1)
            positions = [step * (j + 1) for j in range(len(interleaved_facts))]
            
            # Insert facts at calculated positions
            context_paragraphs = filler_paragraphs.copy()
            for j, position in enumerate(positions):
                if position < len(context_paragraphs):
                    context_paragraphs[position] = interleaved_facts[j]
            
            # Join paragraphs to create the context
            context = '\n\n'.join(context_paragraphs)
            
            # Trim or pad to match target length
            context = self._adjust_text_length(context, context_length)
            
            # Create test cases for each topic in the pair
            for focus_idx, (focus_topic, other_topic) in enumerate([(topic1, topic2), (topic2, topic1)]):
                question = f"Please provide a summary of the key facts about {focus_topic} mentioned in the text. Focus ONLY on {focus_topic}, not on {other_topic}."
                
                relevant_facts = topic1_facts if focus_topic == topic1 else topic2_facts
                irrelevant_facts = topic2_facts if focus_topic == topic1 else topic1_facts
                
                test_id = f"topic_switch_{pair_idx+1}_{focus_idx+1}"
                test_data[test_id] = (context, question, relevant_facts, irrelevant_facts)
        
        return test_data
    
    # Helper methods for generating test data
    def _get_filler_text(self, target_length: int) -> str:
        """
        Get filler text of approximately the target length.
        
        Args:
            target_length: Target length in characters
            
        Returns:
            Filler text
        """
        # Use a public domain text as filler
        try:
            response = requests.get("https://www.gutenberg.org/files/1342/1342-0.txt")
            if response.status_code == 200:
                text = response.text
                
                # Clean up the text
                text = re.sub(r'\r\n', '\n', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                
                # Repeat the text if needed to reach target length
                while len(text) < target_length:
                    text += text
                
                return text[:target_length]
            else:
                logger.warning(f"Failed to fetch filler text: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error fetching filler text: {e}")
        
        # Fallback: generate random text
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", 
                "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", 
                "et", "dolore", "magna", "aliqua"]
        
        paragraphs = []
        total_length = 0
        
        while total_length < target_length:
            # Generate a paragraph
            paragraph_length = random.randint(50, 200)
            paragraph_words = []
            
            for _ in range(paragraph_length):
                paragraph_words.append(random.choice(words))
            
            paragraph = " ".join(paragraph_words)
            paragraphs.append(paragraph)
            
            total_length += len(paragraph) + 2  # +2 for newline chars
        
        return "\n\n".join(paragraphs)
    
    def _get_topic_text(self, topic: str, target_length: int) -> str:
        """
        Generate text about a specific topic.
        
        Args:
            topic: The topic to generate text about
            target_length: Target length in characters
            
        Returns:
            Text about the topic
        """
        # For now, we'll use a simple approach of generating facts about the topic
        facts = self._generate_facts_for_topic(topic, 20)
        
        # Join facts into paragraphs
        paragraphs = []
        for i in range(0, len(facts), 3):
            paragraph = " ".join(facts[i:i+3])
            paragraphs.append(paragraph)
        
        text = "\n\n".join(paragraphs)
        
        # If text is too short, add more facts
        while len(text) < target_length:
            more_facts = self._generate_facts_for_topic(topic, 10)
            more_paragraphs = []
            for i in range(0, len(more_facts), 3):
                paragraph = " ".join(more_facts[i:i+3])
                more_paragraphs.append(paragraph)
            
            text += "\n\n" + "\n\n".join(more_paragraphs)
        
        return self._adjust_text_length(text, target_length)
    
    def _generate_facts_for_topic(self, topic: str, num_facts: int) -> List[str]:
        """
        Generate a list of facts about a specific topic.
        
        Args:
            topic: The topic to generate facts about
            num_facts: Number of facts to generate
            
        Returns:
            List of facts
        """
        # Predefined facts for common topics
        topic_facts = {
            "quantum physics": [
                "Quantum entanglement allows particles to be correlated across vast distances.",
                "Heisenberg's uncertainty principle states that we cannot simultaneously know a particle's position and momentum with perfect precision.",
                "Quantum tunneling enables particles to pass through energy barriers that would be impossible in classical physics.",
                "Quantum superposition allows particles to exist in multiple states simultaneously until measured.",
                "The double-slit experiment demonstrates the wave-particle duality of quantum objects.",
                "Quantum field theory reconciles quantum mechanics with special relativity.",
                "Quantum computing uses qubits that can represent multiple states simultaneously.",
                "The Schrödinger equation describes how quantum states evolve over time.",
                "Quantum decoherence explains how quantum systems interact with their environment.",
                "The Copenhagen interpretation suggests that quantum systems don't have definite properties until measured.",
                "Bell's theorem proves that quantum mechanics cannot be explained by local hidden variables.",
                "The quantum Zeno effect shows that frequent measurements can prevent quantum systems from evolving.",
                "Quantum teleportation allows quantum states to be transmitted over distances.",
                "The Casimir effect demonstrates that vacuum fluctuations can create measurable forces.",
                "Quantum cryptography uses quantum properties to secure communications."
            ],
            "astronomy": [
                "The Milky Way galaxy contains between 100-400 billion stars.",
                "Light from the Andromeda galaxy takes 2.5 million years to reach Earth.",
                "Black holes are regions of spacetime where gravity is so strong that nothing can escape, not even light.",
                "The Sun accounts for 99.86% of the mass in our solar system.",
                "Jupiter has at least 79 moons, the most of any planet in our solar system.",
                "The Great Red Spot on Jupiter is a storm that has been raging for at least 400 years.",
                "Venus rotates in the opposite direction to most planets, a phenomenon known as retrograde rotation.",
                "A day on Mercury lasts approximately 176 Earth days.",
                "The Hubble Space Telescope has made over 1.4 million observations since its launch in 1990.",
                "Neutron stars can rotate up to 600 times per second.",
                "The universe is estimated to be approximately 13.8 billion years old.",
                "The largest known star, UY Scuti, has a radius about 1,700 times that of the Sun.",
                "The asteroid belt between Mars and Jupiter contains millions of asteroids.",
                "Pulsars emit beams of radiation that appear to pulse as the star rotates.",
                "The cosmic microwave background radiation is the afterglow of the Big Bang."
            ],
            "machine learning": [
                "Neural networks are computing systems inspired by the biological neural networks in animal brains.",
                "Deep learning uses multiple layers of neural networks to extract higher-level features from raw input.",
                "Supervised learning algorithms learn from labeled training data to make predictions.",
                "Unsupervised learning algorithms find patterns in unlabeled data.",
                "Reinforcement learning trains algorithms through a reward and punishment system.",
                "Transfer learning applies knowledge from one task to improve learning in another task.",
                "Overfitting occurs when a model learns the training data too well, including its noise and outliers.",
                "Cross-validation helps assess how a model will generalize to independent data.",
                "Gradient descent is an optimization algorithm used to minimize the loss function in machine learning models.",
                "Feature engineering transforms raw data into features that better represent the underlying problem.",
                "Ensemble methods combine multiple learning algorithms to improve overall performance.",
                "The bias-variance tradeoff is a central problem in supervised learning.",
                "Convolutional Neural Networks (CNNs) are particularly effective for image recognition tasks.",
                "Recurrent Neural Networks (RNNs) are designed to recognize patterns in sequences of data.",
                "Generative Adversarial Networks (GANs) consist of two networks that compete against each other."
            ],
            "neuroscience": [
                "The human brain contains approximately 86 billion neurons.",
                "Neurons communicate with each other through electrical and chemical signals.",
                "Synapses are the junctions between neurons where information is transmitted.",
                "Neurotransmitters are chemicals that transmit signals across synapses.",
                "The prefrontal cortex is involved in complex cognitive behaviors and decision making.",
                "The hippocampus plays a crucial role in the formation of new memories.",
                "Neuroplasticity refers to the brain's ability to reorganize itself by forming new neural connections.",
                "The blood-brain barrier protects the brain from harmful substances in the bloodstream.",
                "Mirror neurons fire both when an animal performs an action and when it observes the same action performed by another.",
                "The default mode network is active when the brain is at rest and not focused on the outside world.",
                "Glial cells provide support and protection for neurons in the brain.",
                "The cerebellum coordinates voluntary movements and maintains posture and balance.",
                "The amygdala plays a key role in processing emotions, particularly fear.",
                "The corpus callosum connects the left and right hemispheres of the brain.",
                "Neurogenesis is the process by which new neurons are formed in the brain."
            ],
            "ancient rome": [
                "The Roman Empire at its height covered over 5 million square kilometers.",
                "The Roman Republic was established in 509 BCE after the overthrow of the monarchy.",
                "Julius Caesar was assassinated on the Ides of March (March 15) in 44 BCE.",
                "The Colosseum could hold between 50,000 and 80,000 spectators.",
                "Roman concrete, known as opus caementicium, was more durable than modern concrete.",
                "The Roman calendar initially had 10 months, with January and February added later.",
                "Pompeii was buried under volcanic ash when Mount Vesuvius erupted in 79 CE.",
                "The Roman Senate was originally an advisory council composed of patricians.",
                "The Twelve Tables, created around 450 BCE, were Rome's first written laws.",
                "Roman aqueducts supplied water to cities and towns across the empire.",
                "Latin was the official language of ancient Rome, but Greek was widely spoken in the eastern provinces.",
                "The Pax Romana was a period of relative peace and stability that lasted about 200 years.",
                "Roman citizens wore togas as a sign of their citizenship status.",
                "The Pantheon in Rome has the world's largest unreinforced concrete dome.",
                "The Roman Empire officially fell in 476 CE when Romulus Augustus was deposed."
            ],
            "ancient greece": [
                "Ancient Greek civilization flourished from around 800 BCE to 146 BCE.",
                "The Parthenon was built between 447-432 BCE to honor the goddess Athena.",
                "Democracy originated in Athens around 508 BCE under Cleisthenes.",
                "The Olympic Games began in 776 BCE and were held every four years in Olympia.",
                "Socrates, Plato, and Aristotle established the foundation of Western philosophy.",
                "The Battle of Marathon in 490 BCE was a decisive victory for the Athenians against the Persians.",
                "Greek city-states (polis) were independent states that often competed with each other.",
                "The Peloponnesian War (431-404 BCE) was fought between Athens and Sparta.",
                "Alexander the Great created one of the largest empires in ancient history by the age of 30.",
                "The Greek alphabet was derived from the Phoenician alphabet and is still used today.",
                "Greek mythology featured gods and goddesses who lived on Mount Olympus.",
                "The Iliad and the Odyssey, attributed to Homer, are among the oldest works of Western literature.",
                "Ancient Greeks developed advanced mathematics, including geometry and number theory.",
                "The Oracle of Delphi was consulted for important decisions and prophecies.",
                "Greek architecture featured three column styles: Doric, Ionic, and Corinthian."
            ],
            "renewable energy": [
                "Solar power capacity worldwide has increased more than tenfold in the last decade.",
                "Wind turbines can convert up to 60% of wind energy into electricity, compared to fossil fuel plants which typically achieve 35-40% efficiency.",
                "Hydropower is currently the largest source of renewable electricity, accounting for about 16% of global electricity generation.",
                "Geothermal energy harnesses heat from the Earth's core and can provide continuous baseload power.",
                "Biomass energy is derived from organic materials such as plants, agricultural residues, and waste.",
                "Tidal energy utilizes the natural rise and fall of ocean tides to generate electricity.",
                "The cost of solar photovoltaic modules has decreased by approximately 90% since 2010.",
                "Denmark generated over 50% of its electricity from wind and solar power in 2020.",
                "Energy storage technologies like batteries are crucial for managing the intermittency of renewable sources.",
                "The International Renewable Energy Agency (IRENA) projects that renewable energy could supply 86% of global power demand by 2050.",
                "Floating solar farms are being deployed on reservoirs and lakes to maximize land use efficiency.",
                "Green hydrogen, produced using renewable electricity, is emerging as a key energy carrier for sectors difficult to electrify.",
                "Building-integrated photovoltaics (BIPV) incorporate solar cells directly into building materials.",
                "Virtual power plants aggregate distributed energy resources to provide reliable grid services.",
                "Renewable energy jobs worldwide reached 11.5 million in 2019, with solar photovoltaics being the largest employer."
            ],
            "artificial intelligence": [
                "The term 'artificial intelligence' was coined by John McCarthy in 1956.",
                "Deep Blue became the first computer to defeat a world chess champion when it beat Garry Kasparov in 1997.",
                "Machine learning is a subset of AI that enables systems to learn from data without explicit programming.",
                "Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language.",
                "Computer vision enables machines to interpret and make decisions based on visual data.",
                "The Turing Test, proposed by Alan Turing in 1950, evaluates a machine's ability to exhibit intelligent behavior.",
                "Reinforcement learning was used to train AlphaGo, which defeated world champion Go player Lee Sedol in 2016.",
                "Generative AI can create new content including text, images, music, and video.",
                "Ethical concerns in AI include bias, privacy, accountability, and job displacement.",
                "Explainable AI (XAI) focuses on making AI systems' decisions understandable to humans.",
                "Edge AI processes data locally on devices rather than in the cloud, reducing latency and privacy concerns.",
                "AI alignment research aims to ensure that AI systems act in accordance with human values and intentions.",
                "Federated learning allows models to be trained across multiple devices while keeping data localized.",
                "Neuromorphic computing designs hardware to mimic the structure and function of the human brain.",
                "The AI winter refers to periods of reduced funding and interest in artificial intelligence research."
            ],
            "space exploration": [
                "The first human to journey into space was Yuri Gagarin on April 12, 1961.",
                "The Apollo 11 mission landed the first humans on the Moon on July 20, 1969.",
                "The International Space Station has been continuously occupied since November 2000.",
                "The Voyager 1 spacecraft, launched in 1977, is the most distant human-made object from Earth.",
                "SpaceX became the first private company to send humans to the International Space Station in 2020.",
                "The James Webb Space Telescope, launched in 2021, is the largest and most powerful space telescope ever built.",
                "Mars rovers including Sojourner, Spirit, Opportunity, Curiosity, and Perseverance have explored the Martian surface.",
                "The Artemis program aims to return humans to the Moon and establish sustainable lunar exploration by 2025.",
                "Exoplanets are planets that orbit stars outside our solar system, with over 5,000 confirmed discoveries.",
                "The Hubble Space Telescope has made over 1.4 million observations and generated more than 150 terabytes of data.",
                "The Parker Solar Probe is designed to study the Sun's corona by flying closer to the Sun than any previous spacecraft.",
                "The search for extraterrestrial life focuses on habitable zones around stars and ocean worlds like Europa and Enceladus.",
                "Space debris, consisting of defunct satellites and mission-related debris, poses a growing threat to space operations.",
                "The Gateway is a planned lunar orbital outpost that will serve as a staging point for lunar and Mars missions.",
                "Reusable rocket technology has significantly reduced the cost of access to space in the past decade."
            ]
        }
        
        # If we have predefined facts for this topic, use them
        if topic in topic_facts:
            # Shuffle the facts to get a random selection
            facts = topic_facts[topic].copy()
            random.shuffle(facts)
            return facts[:num_facts]
        
        # Otherwise, generate generic facts about the topic
        facts = []
        for i in range(num_facts):
            fact = f"Fact {i+1} about {topic}: This is a placeholder for a fact about {topic}."
            facts.append(fact)
        
        return facts
    
    def _adjust_text_length(self, text: str, target_length: int) -> str:
        """
        Adjust text to match the target length.
        
        Args:
            text: Text to adjust
            target_length: Target length in characters
            
        Returns:
            Adjusted text
        """
        if len(text) > target_length:
            # Trim the text
            return text[:target_length]
        elif len(text) < target_length:
            # Pad the text with spaces
            return text + " " * (target_length - len(text))
        else:
            return text
    
    # Response evaluation methods
    def _evaluate_retrieval_response(self, response: str, expected_answer: str) -> Tuple[bool, Dict]:
        """
        Evaluate a response for the basic retrieval test.
        
        Args:
            response: Model's response
            expected_answer: Expected answer
            
        Returns:
            Tuple of (is_correct, match_details)
        """
        # Extract key information from the expected answer
        key_entities = re.findall(r'\b[A-Z][a-zA-Z]*\b', expected_answer)
        key_numbers = re.findall(r'\b\d+\b', expected_answer)
        
        # Check for key entities in the response
        entity_matches = {}
        for entity in key_entities:
            entity_matches[entity] = entity.lower() in response.lower()
        
        # Check for key numbers in the response
        number_matches = {}
        for number in key_numbers:
            number_matches[number] = number in response
        
        # Extract key facts from the expected answer
        key_facts = self._extract_key_facts(expected_answer)
        
        # Check for key facts in the response
        fact_matches = {}
        for fact in key_facts:
            # Consider a fact matched if most of its significant words are present
            words = fact.lower().split()
            significant_words = [w for w in words if len(w) > 3 and w not in ["with", "from", "have", "been", "that", "this", "their", "about"]]
            
            if not significant_words:  # If no significant words, use all words
                significant_words = words
                
            # Count how many significant words are present
            words_present = sum(1 for word in significant_words if word in response.lower())
            match_ratio = words_present / len(significant_words) if significant_words else 0
            
            fact_matches[fact] = match_ratio >= 0.6  # Lower threshold for better recall
        
        # Calculate overall match score
        entity_score = sum(entity_matches.values()) / len(entity_matches) if entity_matches else 1.0
        number_score = sum(number_matches.values()) / len(number_matches) if number_matches else 1.0
        fact_score = sum(fact_matches.values()) / len(fact_matches) if fact_matches else 1.0
        
        # Weighted average of scores
        overall_score = (0.3 * entity_score) + (0.3 * number_score) + (0.4 * fact_score)
        
        # Prepare match details
        match_details = {
            "entity_matches": entity_matches,
            "number_matches": number_matches,
            "fact_matches": fact_matches,
            "entity_score": entity_score,
            "number_score": number_score,
            "fact_score": fact_score,
            "overall_score": overall_score
        }
        
        # Consider correct if overall score is at least 0.7
        is_correct = overall_score >= 0.7
        
        return is_correct, match_details
    
    def _evaluate_association_response(self, response: str, expected_answer: str) -> Tuple[bool, Dict]:
        """
        Evaluate a response for the memory association test.
        
        Args:
            response: Model's response
            expected_answer: Expected answer
            
        Returns:
            Tuple of (is_correct, match_details)
        """
        # Extract key facts from the expected answer
        key_facts = self._extract_key_facts(expected_answer)
        
        # Check for key facts in the response
        fact_matches = {}
        for fact in key_facts:
            # Consider a fact matched if most of its significant words are present
            words = fact.lower().split()
            significant_words = [w for w in words if len(w) > 3 and w not in ["with", "from", "have", "been", "that", "this", "their", "about"]]
            
            if not significant_words:  # If no significant words, use all words
                significant_words = words
                
            # Count how many significant words are present
            words_present = sum(1 for word in significant_words if word in response.lower())
            match_ratio = words_present / len(significant_words) if significant_words else 0
            
            fact_matches[fact] = match_ratio >= 0.6  # Lower threshold for better recall
        
        # Calculate overall match score
        fact_score = sum(fact_matches.values()) / len(fact_matches) if fact_matches else 0
        
        # Prepare match details
        match_details = {
            "fact_matches": fact_matches,
            "fact_score": fact_score
        }
        
        # Consider correct if fact score is at least 0.7
        is_correct = fact_score >= 0.7
        
        return is_correct, match_details
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """
        Extract key facts from a text.
        
        Args:
            text: Text to extract facts from
            
        Returns:
            List of key facts
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Remove empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # For short texts, consider each sentence a fact
        if len(sentences) <= 5:
            return sentences
        
        # For longer texts, try to group sentences into coherent facts
        facts = []
        current_fact = ""
        
        for sentence in sentences:
            if len(current_fact) + len(sentence) <= 150:  # Arbitrary length limit for a fact
                if current_fact:
                    current_fact += " " + sentence
                else:
                    current_fact = sentence
            else:
                if current_fact:
                    facts.append(current_fact)
                current_fact = sentence
        
        if current_fact:
            facts.append(current_fact)
        
        return facts
    
    def _evaluate_multi_document_response(self, response: str, key_points: List[str]) -> Tuple[float, Dict]:
        """
        Evaluate a response for the multi-document summarization test.
        
        Args:
            response: Model's response
            key_points: List of key points that should be included in the summary
            
        Returns:
            Tuple of (score, evaluation_details)
        """
        # Check how many key points are mentioned in the response
        points_mentioned = []
        points_missed = []
        match_scores = {}
        
        for point in key_points:
            # Check for direct match
            if point.lower() in response.lower():
                points_mentioned.append(point)
                match_scores[point] = 1.0
                continue
                
            # Check for partial match
            words = point.lower().split()
            significant_words = [w for w in words if len(w) > 3 and w not in ["with", "from", "have", "been", "that", "this", "their", "about"]]
            
            if not significant_words:  # If no significant words, use all words
                significant_words = words
                
            # Count how many significant words are present
            words_present = sum(1 for word in significant_words if word in response.lower())
            match_ratio = words_present / len(significant_words) if significant_words else 0
            
            match_scores[point] = match_ratio
            
            if match_ratio >= 0.6:  # Lower threshold for better recall
                points_mentioned.append(point)
            else:
                points_missed.append(point)
        
        # Calculate the score
        score = len(points_mentioned) / len(key_points) if key_points else 0
        
        # Prepare evaluation details
        details = {
            "points_mentioned": points_mentioned,
            "points_missed": points_missed,
            "match_scores": match_scores,
            "total_points": len(key_points),
            "points_mentioned_count": len(points_mentioned),
            "recall_score": score
        }
        
        return score, details
    
    def _evaluate_topic_switching_response(self, response: str, relevant_facts: List[str], irrelevant_facts: List[str]) -> Tuple[float, Dict]:
        """
        Evaluate a response for the topic switching test.
        
        Args:
            response: Model's response
            relevant_facts: Facts that should be included in the response
            irrelevant_facts: Facts that should not be included in the response
            
        Returns:
            Tuple of (score, evaluation_details)
        """
        # Check how many relevant facts are mentioned
        relevant_mentioned = []
        relevant_missed = []
        relevant_match_scores = {}
        
        for fact in relevant_facts:
            match_ratio = self._calculate_fact_match_ratio(fact, response)
            relevant_match_scores[fact] = match_ratio
            
            if match_ratio >= 0.6:  # Lower threshold for better recall
                relevant_mentioned.append(fact)
            else:
                relevant_missed.append(fact)
        
        # Check how many irrelevant facts are mentioned (these should be avoided)
        irrelevant_mentioned = []
        irrelevant_match_scores = {}
        
        for fact in irrelevant_facts:
            match_ratio = self._calculate_fact_match_ratio(fact, response)
            irrelevant_match_scores[fact] = match_ratio
            
            if match_ratio >= 0.6:
                irrelevant_mentioned.append(fact)
        
        # Calculate the score
        # We want to reward mentioning relevant facts and penalize mentioning irrelevant facts
        recall = len(relevant_mentioned) / len(relevant_facts) if relevant_facts else 0
        precision = 1 - (len(irrelevant_mentioned) / len(irrelevant_facts) if irrelevant_facts else 0)
        
        # Combine recall and precision into a single score
        # We weight recall more heavily since that's the primary goal
        score = (0.7 * recall) + (0.3 * precision)
        
        # Prepare evaluation details
        details = {
            "relevant_mentioned": relevant_mentioned,
            "relevant_missed": relevant_missed,
            "irrelevant_mentioned": irrelevant_mentioned,
            "relevant_match_scores": relevant_match_scores,
            "irrelevant_match_scores": irrelevant_match_scores,
            "recall": recall,
            "precision": precision,
            "combined_score": score
        }
        
        return score, details
    
    def _calculate_fact_match_ratio(self, fact: str, text: str) -> float:
        """
        Calculate the match ratio between a fact and a text.
        
        Args:
            fact: The fact to check for
            text: The text to search in
            
        Returns:
            Match ratio between 0 and 1
        """
        # Extract key entities and phrases from the fact
        words = fact.lower().split()
        significant_words = [w for w in words if len(w) > 3 and w not in ["with", "from", "have", "been", "that", "this", "their", "about"]]
        
        if not significant_words:  # If no significant words, use all words
            significant_words = words
            
        # Count how many significant words are present
        words_present = sum(1 for word in significant_words if word in text.lower())
        
        # Calculate match ratio
        return words_present / len(significant_words) if significant_words else 0
    
    def _is_fact_mentioned(self, fact: str, text: str) -> bool:
        """
        Check if a fact is mentioned in the text.
        
        Args:
            fact: The fact to check for
            text: The text to search in
            
        Returns:
            True if the fact is mentioned, False otherwise
        """
        match_ratio = self._calculate_fact_match_ratio(fact, text)
        return match_ratio >= 0.6  # Lower threshold for better recall

def main():
    parser = argparse.ArgumentParser(description="Ollama LLM Context Length Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Name of the Ollama model to benchmark")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save benchmark results")
    parser.add_argument("--context-lengths", type=str, default="1000,5000,10000,20000", 
                        help="Comma-separated list of context lengths to test")
    parser.add_argument("--test", type=str, default="all", 
                        choices=["all", "basic_retrieval", "memory_association", "multi_document", "topic_switching"],
                        help="Which test to run")
    parser.add_argument("--test-mode", action="store_true", 
                        help="Run in test mode without requiring Ollama (for testing the script)")
    
    args = parser.parse_args()
    
    # Parse context lengths
    context_lengths = [int(length) for length in args.context_lengths.split(",")]
    
    # Create benchmark instance with model-specific output directory
    # Use os.path.join for cross-platform compatibility
    sanitized_model_name = re.sub(r'[\\/*?:"<>|]', '_', args.model)
    model_output_dir = os.path.join(args.output_dir, sanitized_model_name)
    
    benchmark = ContextBenchmark(args.model, model_output_dir, args.test_mode)
    
    # Run the specified test(s)
    if args.test == "all":
        benchmark.run_all_tests(context_lengths)
    elif args.test == "basic_retrieval":
        benchmark.run_basic_retrieval_test(context_lengths)
    elif args.test == "memory_association":
        benchmark.run_memory_association_test(context_lengths)
    elif args.test == "multi_document":
        benchmark.run_multi_document_test(context_lengths)
    elif args.test == "topic_switching":
        benchmark.run_topic_switching_test(context_lengths)

if __name__ == "__main__":
    main()
