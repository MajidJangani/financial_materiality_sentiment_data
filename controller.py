"""
ESG MATERIALITY ANALYSIS CONTROLLER
====================================
High-level orchestration for ESG materiality pipeline with PostgreSQL integration.

1. Connect to PostgreSQL database for ESG data
2. Orchestrate the complete analysis pipeline
3. Handle errors gracefully
4. Save results in multiple formats (JSON, CSV)
5. Run batch analysis on multiple entities

Author: Majid Jangani
Book: ESG Financial Materiality Assessment
Version: 2.0 (Book Publication Edition)
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extensions import connection as PgConnection
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

# Import our analysis engine
import engine

warnings.filterwarnings('ignore')


# =================================================================
# SECTION 1: DATABASE CONFIGURATION
# =================================================================

class DatabaseConfig:
    """
    Database configuration management.
    

    In production, you would load these from environment variables
    or a secure configuration file, never hardcode credentials!
    
    Example usage:
        >>> db_config = DatabaseConfig(
        >>>     host="localhost",
        >>>     database="esg_database",
        >>>     user="your_username",
        >>>     password="your_password"
        >>> )
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 database: str = "esg_database",
                 user: str = "postgres",
                 password: str = "your_password",
                 port: int = 5432):
        """
        Initialize database configuration.
        
        Args:
            host: PostgreSQL server hostname
            database: Database name
            user: Database username
            password: Database password
            port: PostgreSQL port (default: 5432)
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
    
    @classmethod
    def from_env(cls):
        """
        Load configuration from environment variables.
        
        Set environment variables before running:
        
        export DB_HOST=localhost
        export DB_NAME=esg_database
        export DB_USER=postgres
        export DB_PASSWORD=your_password
        export DB_PORT=5432
        
        Example:
            >>> db_config = DatabaseConfig.from_env()
        """
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "esg_database"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=int(os.getenv("DB_PORT", "5432"))
        )
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string."""
        return f"host={self.host} dbname={self.database} user={self.user} password={self.password} port={self.port}"


# =================================================================
# SECTION 2: THE MAIN CONTROLLER CLASS
# =================================================================

class ESGMaterialityController:
    """
    Main controller for ESG materiality analysis pipeline.
    
    1. Connects to database
    2. Loads entity data
    3. Runs Framework 1 (feature selection)
    4. Runs Framework 2 (temporal validation)
    5. Runs regime detection
    6. Saves results
    
    Example usage:
        >>> controller = ESGMaterialityController(db_config)
        >>> results = controller.run_analysis(entity_id=1)
        >>> controller.save_results(results, "output/entity_1_results.json")
    """
    
    def __init__(self, db_config: DatabaseConfig, output_dir: str = "output"):
        """
        Initialize the ESG Materiality Controller.
        
        Args:
            db_config: Database configuration object
            output_dir: Directory for saving results
        """
        self.db_config = db_config
        self.output_dir = output_dir
        self.connection = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "json_results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv_exports"), exist_ok=True)
        
        print(f"✓ ESG Materiality Controller initialized")
        print(f"  Output directory: {output_dir}")
    
    # -----------------------------------------------------------------
    # Database Connection Methods
    # -----------------------------------------------------------------
    
    def connect(self) -> PgConnection:
        """
        Establish connection to PostgreSQL database.
        
        TUTORIAL: This method creates a database connection that will be
        reused for multiple queries. Always call disconnect() when done!
        
        Returns:
            PostgreSQL connection object
            
        Raises:
            psycopg2.Error: If connection fails
            
        Example:
            >>> controller.connect()
            >>> # ... do work ...
            >>> controller.disconnect()
        """
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.host,
                database=self.db_config.database,
                user=self.db_config.user,
                password=self.db_config.password,
                port=self.db_config.port
            )
            print(f"✓ Connected to database: {self.db_config.database}")
            return self.connection
            
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """
        Close database connection.
        
        TUTORIAL: Always close connections when done to free resources.
        """
        if self.connection:
            self.connection.close()
            self.connection = None
            print(f"✓ Database connection closed")
    
    # -----------------------------------------------------------------
    #  Data Loading Methods
    # -----------------------------------------------------------------
    
    def load_entity_data(self, entity_id: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Load ESG data for a single entity from PostgreSQL.
        
        1. Query database schema to find available columns
        2. Build dynamic SQL queries
        3. Load data into pandas DataFrame
        4. Perform data quality validation
        
        Args:
            entity_id: Unique entity identifier
            
        Returns:
            Tuple of (entity_dataframe, data_quality_metrics)
            
        Raises:
            ValueError: If data quality is insufficient
            psycopg2.Error: If database query fails
            
        Example:
            >>> controller.connect()
            >>> entity_data, quality = controller.load_entity_data(entity_id=1)
            >>> print(f"Loaded {len(entity_data)} rows with {quality['pbs_features']} PBS features")
        """
        
        if not self.connection:
            raise RuntimeError("Must call connect() before loading data")
        
        print(f"\n Loading data for Entity {entity_id}...")
        
        try:
            # Step 1: Get available columns from database schema
            schema_query = """
            SELECT column_name, data_type
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position;
            """
            
            columns_df = pd.read_sql_query(
                schema_query, 
                self.connection, 
                params=[str(entity_id)]
            )
            
            available_columns = columns_df['column_name'].tolist()
            
            # Step 2: Find PBS columns (sentiment features)
            pbs_columns = [col for col in available_columns if col.startswith('pbs_')]
            pbs_columns.sort(key=lambda x: int(x.split('_')[1]))
            
            print(f"  Found {len(pbs_columns)} PBS features")
            
            # Step 3: Build query for required columns
            required_columns = ['date', 'close_price'] + pbs_columns
            
            # Add volume if available (needed for some analyses)
            if 'volume' in available_columns:
                required_columns.append('volume')
                print(f"  ✓ Volume data available")
            else:
                print(f"  ⚠ Volume data not available")
            
            # Create SQL SELECT clause
            select_clause = ', '.join([f'"{col}"' for col in required_columns])
            
            # Step 4: Execute query with data quality filters
            query = f"""
            SELECT {select_clause}
            FROM "{entity_id}"
            WHERE date IS NOT NULL 
              AND close_price IS NOT NULL 
              AND close_price > 0
            ORDER BY date ASC;
            """
            
            df = pd.read_sql_query(query, self.connection)
            
            # Step 5: Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Step 6: Data quality assessment
            data_quality = {
                'entity_id': entity_id,
                'total_rows': len(df),
                'date_range': (df['date'].min(), df['date'].max()),
                'pbs_features': len(pbs_columns),
                'has_volume': 'volume' in df.columns,
                'missing_data_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
                'price_stats': {
                    'min': df['close_price'].min(),
                    'max': df['close_price'].max(),
                    'mean': df['close_price'].mean(),
                    'std': df['close_price'].std()
                }
            }
            
            # Step 7: Validate data quality
            if data_quality['total_rows'] < 100:
                raise ValueError(
                    f"Insufficient data: {data_quality['total_rows']} rows (minimum 100 required)"
                )
            
            if data_quality['pbs_features'] < 15:
                raise ValueError(
                    f"Insufficient PBS features: {data_quality['pbs_features']} (minimum 15 required)"
                )
            
            print(f"  ✓ Data quality check passed")
            print(f"    Rows: {data_quality['total_rows']}")
            print(f"    Date range: {data_quality['date_range'][0].date()} to {data_quality['date_range'][1].date()}")
            print(f"    PBS features: {data_quality['pbs_features']}")
            
            return df, data_quality
            
        except psycopg2.Error as e:
            print(f"  ✗ Database query failed: {e}")
            raise
        except Exception as e:
            print(f"  ✗ Data loading failed: {e}")
            raise
    
    # -----------------------------------------------------------------
    #  Main Analysis Orchestration
    # -----------------------------------------------------------------
    
    def run_analysis(self, entity_id: int, config: Optional[Dict] = None) -> Dict:
        """
        Execute complete ESG materiality analysis pipeline.
        
        1. Loads entity data from database
        2. Runs Framework 1 (27 → 12 → 6 feature selection)
        3. Runs Framework 2 (26-week temporal validation)
        4. Runs regime detection (pillar-level and driver-level)
        5. Compiles comprehensive results
        
        This method demonstrates how to call engine functions in the correct
        order and handle errors gracefully.
        
        Args:
            entity_id: Unique entity identifier
            config: Optional configuration dict (uses DEFAULT_CONFIG if None)
            
        Returns:
            Dict with complete analysis results
            
        Example:
            >>> controller.connect()
            >>> results = controller.run_analysis(entity_id=1)
            >>> print(f"Status: {results['status']}")
            >>> controller.disconnect()
        """
        
        if config is None:
            config = engine.DEFAULT_CONFIG
        
        print(f"\n{'='*70}")
        print(f" ESG MATERIALITY ANALYSIS - ENTITY {entity_id}")
        print(f"{'='*70}")
        
        results = {
            'entity_id': entity_id,
            'timestamp': datetime.now().isoformat(),
            'config_used': config,
            'status': 'IN_PROGRESS'
        }
        
        try:
            # ============================================================
            # STEP 1: LOAD AND VALIDATE DATA
            # ============================================================
            entity_data, data_quality = self.load_entity_data(entity_id)
            results['data_quality'] = data_quality
            
            # Get PBS feature columns
            pbs_features = [col for col in entity_data.columns if col.startswith('pbs_')]
            pbs_features.sort(key=lambda x: int(x.split('_')[1]))
            
            # ============================================================
            # STEP 2: FRAMEWORK 1 - FEATURE SELECTION (27 → 12 → 6)
            # ============================================================
            print(f"\n{'='*70}")
            print(f" FRAMEWORK 1: FEATURE SELECTION (27 → 12 → 6)")
            print(f"{'='*70}")
            
            # Calculate log returns for target creation
            returns = np.log(
                entity_data['close_price'] / entity_data['close_price'].shift(1)
            ).dropna()
            
            # Create BOTH binary and 3-class targets
            print(f"\n Creating target variables...")
            targets_binary, metadata_binary = engine.create_research_enhanced_binary_targets(
                returns, verbose=True, config=config.get('target_creation')
            )
            
            targets_3class, metadata_3class, quality_gates = engine.create_three_class_targets_skewness_aware(
                returns, verbose=True, config=config.get('target_creation')
            )
            
            # Prepare training data (align with targets)
            train_df = entity_data.copy()
            train_df = train_df.iloc[-len(targets_3class):].reset_index(drop=True)
            
            # Setup cross-validation
            print(f"\n Setting up cross-validation...")
            cv_splits = engine.setup_conservative_cv(
                train_df[pbs_features],
                targets_binary,
                config.get('framework1', {})
            )
            
            # Stage 1: Statistical Screening (27 → 12)
            print(f"\n Stage 1: Statistical Screening (27 → 12)...")
            importance_results = engine.calculate_unified_importance_scores_with_binary(
                train_df[pbs_features],
                targets_binary,
                targets_3class,
                cv_splits,
                config.get('framework1', {})
            )
            
            composite_scores = engine.create_enhanced_composite_scores(
                importance_results,
                config.get('framework1', {})
            )
            
            selected_drivers_12, pillar_counts_12 = engine.select_balanced_drivers_enhanced(
                composite_scores,
                target_features=12
            )
            
            # Stage 2: RF Refinement (12 → 6)
            print(f"\n Stage 2: Random Forest Refinement (12 → 6)...")
            selected_features_12 = [d['feature'] for d in selected_drivers_12]
            X_selected_12 = train_df[selected_features_12]
            
            final_drivers, cut_drivers, rf_model = engine.refine_features_with_rf(
                X_selected_12=X_selected_12,
                y=targets_binary,
                selected_drivers_12=selected_drivers_12,
                target_final=6,
                verbose=True
            )
            
            # Extract final features
            selected_features = [d['feature'] for d in final_drivers]
            pillar_counts_final = {}
            for driver in final_drivers:
                pillar = driver['pillar']
                pillar_counts_final[pillar] = pillar_counts_final.get(pillar, 0) + 1
            
            # Store Framework 1 results
            results['framework1_results'] = {
                'selected_drivers': final_drivers,
                'selected_features': selected_features,
                'selected_drivers_12': selected_drivers_12,
                'cut_drivers': cut_drivers,
                'pillar_distribution': pillar_counts_final,
                'importance_results': {
                    'MI': [{'feature': f, 'score': s} for f, s, _ in importance_results['MI']],
                    'AUC': [{'feature': f, 'score': s} for f, s, _ in importance_results['AUC']]
                },
                'target_metadata': {
                    'binary': metadata_binary,
                    'three_class': metadata_3class,
                    'quality_gates': quality_gates
                }
            }
            
            print(f"\n✓ Framework 1 Complete")
            print(f"  Final drivers: {[d['driver_number'] for d in final_drivers]}")
            print(f"  Pillar distribution: {pillar_counts_final}")
            
            # ============================================================
            # STEP 3: FRAMEWORK 2 - TEMPORAL VALIDATION (26-week)
            # ============================================================
            print(f"\n{'='*70}")
            print(f" FRAMEWORK 2: TEMPORAL VALIDATION (26-week windows)")
            print(f"{'='*70}")
            
            walkforward_results, window_size, step_size = engine.run_walkforward_analysis_comprehensive_original(
                entity_data,
                selected_features,
                config.get('framework2', {})
            )
            
            print(f"\n✓ Framework 2 Complete")
            print(f"  Analyzed {len(walkforward_results)} temporal windows")
            
            # ============================================================
            # STEP 4: REGIME DETECTION
            # ============================================================
            print(f"\n{'='*70}")
            print(f" REGIME DETECTION")
            print(f"{'='*70}")
            
            # Pillar-level regime detection
            print(f"\n🔍 Level 1: Pillar-level regime detection...")
            pillar_regimes, pillar_regime_list = engine.detect_pillar_level_regimes(
                walkforward_results,
                config.get('framework2', {})
            )
            
            # Individual driver regime tracking
            print(f"\n🔍 Level 2: Individual driver regime tracking...")
            individual_regimes = engine.track_individual_driver_regimes(
                walkforward_results,
                config.get('framework2', {})
            )
            
            # Calculate stability metrics
            print(f"\n📈 Calculating driver stability metrics...")
            driver_stability = engine.calculate_driver_stability_metrics(
                walkforward_results,
                individual_regimes
            )
            
            # Store Framework 2 results
            results['framework2_results'] = {
                'walkforward_results': walkforward_results,
                'window_parameters': {
                    'window_size': window_size,
                    'step_size': step_size,
                    'num_windows': len(walkforward_results)
                },
                'pillar_regimes': pillar_regimes,
                'pillar_regime_list': pillar_regime_list,
                'individual_regimes': individual_regimes,
                'driver_stability': driver_stability
            }
            
            print(f"\n✓ Regime Detection Complete")
            print(f"  Current regime: {pillar_regimes['current_regime']}")
            print(f"  Total regimes detected: {pillar_regimes['total_regimes']}")
            
            # ============================================================
            # STEP 5: COMPILE FINAL RESULTS
            # ============================================================
            results['status'] = 'SUCCESS'
            results['summary'] = self._create_summary(results)
            
            print(f"\n{'='*70}")
            print(f"✅ ANALYSIS COMPLETE - ENTITY {entity_id}")
            print(f"{'='*70}")
            
            return results
            
        except Exception as e:
            print(f"\n✗ Analysis failed for Entity {entity_id}: {e}")
            results['status'] = 'FAILED'
            results['error'] = str(e)
            import traceback
            results['traceback'] = traceback.format_exc()
            return results
    
    def _create_summary(self, results: Dict) -> Dict:
        """Create executive summary from analysis results."""
        
        fw1 = results.get('framework1_results', {})
        fw2 = results.get('framework2_results', {})
        
        # Extract key insights
        final_drivers = fw1.get('selected_drivers', [])
        pillar_regimes = fw2.get('pillar_regimes', {})
        driver_stability = fw2.get('driver_stability', {})
        
        # Top stable drivers
        stable_drivers = sorted(
            driver_stability.items(),
            key=lambda x: x[1].get('combined_stability', 0),
            reverse=True
        )[:3]
        
        # Emerging drivers
        emerging_drivers = [
            d for d, info in driver_stability.items() 
            if info.get('is_emerging', False)
        ]
        
        summary = {
            'entity_id': results['entity_id'],
            'top_material_drivers': [d['driver_number'] for d in final_drivers],
            'top_stable_drivers': [d[0] for d in stable_drivers],
            'emerging_drivers': emerging_drivers,
            'current_esg_regime': pillar_regimes.get('current_regime', 'Unknown'),
            'regime_stability': pillar_regimes.get('regime_stability', 0),
            'windows_analyzed': fw2.get('window_parameters', {}).get('num_windows', 0)
        }
        
        return summary
    
    # -----------------------------------------------------------------
    # Results Export Methods
    # -----------------------------------------------------------------
    
    def save_results(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Save analysis results to JSON file.
        
        1. Convert numpy/pandas types to JSON-serializable formats
        2. Create organized output directory structure
        3. Generate timestamped filenames
        
        Args:
            results: Analysis results dictionary
            output_path: Optional custom output path
            
        Returns:
            Path to saved JSON file
            
        Example:
            >>> json_file = controller.save_results(results)
            >>> print(f"Results saved to: {json_file}")
        """
        
        if output_path is None:
            entity_id = results['entity_id']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, 
                "json_results",
                f"entity_{entity_id}_results_{timestamp}.json"
            )
        
        # Convert numpy/pandas types for JSON serialization
        results_clean = self._convert_for_json(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        
        return output_path
    
    def export_to_csv(self, results: Dict) -> Dict[str, str]:
        """
        Export analysis results to CSV files.
        
        1. framework1_results.csv - Selected drivers with scores
        2. temporal_analysis.csv - Window-by-window results
        3. regime_analysis.csv - Regime detection results
        4. driver_stability.csv - Stability metrics
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dict mapping file types to file paths
            
        Example:
            >>> csv_files = controller.export_to_csv(results)
            >>> print(f"Framework 1 results: {csv_files['framework1']}")
        """
        
        if results['status'] != 'SUCCESS':
            print(f"⚠ Cannot export CSV - analysis failed")
            return {}
        
        entity_id = results['entity_id']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = os.path.join(self.output_dir, "csv_exports")
        
        exported_files = {}
        
        # CSV 1: Framework 1 Results
        fw1 = results.get('framework1_results', {})
        if fw1.get('selected_drivers'):
            fw1_data = []
            for i, driver in enumerate(fw1['selected_drivers'], 1):
                fw1_data.append({
                    'rank': i,
                    'driver_number': driver['driver_number'],
                    'pillar': driver['pillar'],
                    'mi_score': driver.get('mi_score', 0),
                    'auc_score': driver.get('auc_score', 0),
                    'rf_importance': driver.get('rf_importance', 0),
                    'composite_score': driver.get('composite_score', 0)
                })
            
            df_fw1 = pd.DataFrame(fw1_data)
            fw1_file = os.path.join(csv_dir, f"entity_{entity_id}_framework1_{timestamp}.csv")
            df_fw1.to_csv(fw1_file, index=False)
            exported_files['framework1'] = fw1_file
            print(f"  ✓ Framework 1 results: {fw1_file}")
        
        # CSV 2: Temporal Analysis
        fw2 = results.get('framework2_results', {})
        walkforward_results = fw2.get('walkforward_results', [])
        if walkforward_results:
            temporal_data = []
            for window in walkforward_results:
                for feature_info in window['feature_rankings']:
                    temporal_data.append({
                        'window_id': window['window_id'],
                        'period_start': window['period_start'],
                        'period_end': window['period_end'],
                        'driver_number': feature_info['driver_number'],
                        'pillar': feature_info['pillar'],
                        'rank': feature_info['rank'],
                        'combined_score': feature_info['combined_score'],
                        'mi_score': feature_info['mi_score']
                    })
            
            df_temporal = pd.DataFrame(temporal_data)
            temporal_file = os.path.join(csv_dir, f"entity_{entity_id}_temporal_{timestamp}.csv")
            df_temporal.to_csv(temporal_file, index=False)
            exported_files['temporal'] = temporal_file
            print(f"  ✓ Temporal analysis: {temporal_file}")
        
        # CSV 3: Regime Analysis
        pillar_regimes = fw2.get('pillar_regimes', {})
        pillar_regime_list = fw2.get('pillar_regime_list', [])
        if pillar_regime_list:
            regime_data = []
            for regime in pillar_regime_list:
                regime_data.append({
                    'regime_id': regime['regime_id'],
                    'pillar': regime['pillar'],
                    'window_start': regime['regime_start'],
                    'window_end': regime['regime_end'],
                    'duration_windows': regime['duration_windows'],
                    'period_start': regime['period_start'],
                    'period_end': regime['period_end']
                })
            
            df_regimes = pd.DataFrame(regime_data)
            regime_file = os.path.join(csv_dir, f"entity_{entity_id}_regimes_{timestamp}.csv")
            df_regimes.to_csv(regime_file, index=False)
            exported_files['regimes'] = regime_file
            print(f"  ✓ Regime analysis: {regime_file}")
        
        # CSV 4: Driver Stability
        driver_stability = fw2.get('driver_stability', {})
        if driver_stability:
            stability_data = []
            for driver_num, metrics in driver_stability.items():
                stability_data.append({
                    'driver_number': driver_num,
                    'pillar': metrics['pillar'],
                    'avg_rank': metrics['avg_rank'],
                    'combined_stability': metrics['combined_stability'],
                    'current_regime': metrics['current_regime'],
                    'is_emerging': metrics['is_emerging'],
                    'is_declining': metrics['is_declining']
                })
            
            df_stability = pd.DataFrame(stability_data)
            stability_file = os.path.join(csv_dir, f"entity_{entity_id}_stability_{timestamp}.csv")
            df_stability.to_csv(stability_file, index=False)
            exported_files['stability'] = stability_file
            print(f"  ✓ Driver stability: {stability_file}")
        
        print(f"\n✓ Exported {len(exported_files)} CSV files")
        
        return exported_files
    
    def _convert_for_json(self, obj):
        """
        Convert numpy/pandas types to JSON-serializable formats.
        
        JSON can't handle numpy arrays, pandas timestamps, etc.
        This method recursively converts them to standard Python types.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(v) for v in obj]
        else:
            return obj
    
    # -----------------------------------------------------------------
    # Batch Processing Methods
    # -----------------------------------------------------------------
    
    def run_batch_analysis(self, entity_ids: List[int], config: Optional[Dict] = None) -> Dict:
        """
        Run analysis on multiple entities in batch.
        
        1. Process multiple entities efficiently
        2. Handle errors for individual entities without stopping batch
        3. Collect summary statistics across entities
        4. Generate batch reports
        
        Args:
            entity_ids: List of entity IDs to analyze
            config: Optional configuration dict
            
        Returns:
            Dict with batch results and summary statistics
            
        Example:
            >>> controller.connect()
            >>> batch_results = controller.run_batch_analysis([1, 4, 24, 126])
            >>> print(f"Success rate: {batch_results['success_rate']:.1%}")
            >>> controller.disconnect()
        """
        
        print(f"\n{'='*70}")
        print(f" BATCH ANALYSIS: {len(entity_ids)} entities")
        print(f"{'='*70}")
        
        batch_results = {
            'start_time': datetime.now().isoformat(),
            'entity_ids': entity_ids,
            'results': {},
            'summary': {}
        }
        
        successful = 0
        failed = 0
        
        for i, entity_id in enumerate(entity_ids, 1):
            print(f"\n{'─'*70}")
            print(f"Processing {i}/{len(entity_ids)}: Entity {entity_id}")
            print(f"{'─'*70}")
            
            try:
                # Run analysis
                result = self.run_analysis(entity_id, config)
                
                # Save results
                json_file = self.save_results(result)
                csv_files = self.export_to_csv(result)
                
                # Store in batch results
                batch_results['results'][entity_id] = {
                    'status': result['status'],
                    'summary': result.get('summary', {}),
                    'json_file': json_file,
                    'csv_files': csv_files
                }
                
                if result['status'] == 'SUCCESS':
                    successful += 1
                    print(f"✅ Entity {entity_id}: SUCCESS")
                else:
                    failed += 1
                    print(f" Entity {entity_id}: FAILED - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed += 1
                print(f" Entity {entity_id}: CRASHED - {e}")
                batch_results['results'][entity_id] = {
                    'status': 'CRASHED',
                    'error': str(e)
                }
        
        # Compile batch summary
        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['summary'] = {
            'total_entities': len(entity_ids),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(entity_ids) if entity_ids else 0
        }
        
        # Save batch report
        batch_report_file = os.path.join(
            self.output_dir,
            f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(batch_report_file, 'w') as f:
            json.dump(self._convert_for_json(batch_results), f, indent=2)
        
        print(f"\n{'='*70}")
        print(f" BATCH ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"Success rate: {successful}/{len(entity_ids)} ({batch_results['summary']['success_rate']:.1%})")
        print(f"Batch report: {batch_report_file}")
        
        return batch_results
    
    # -----------------------------------------------------------------
    # Context Manager Support 
    # -----------------------------------------------------------------
    
    def __enter__(self):
        """
        Enable 'with' statement usage.
        
        with ESGMaterialityController(db_config) as controller:
            results = controller.run_analysis(1)
        # Connection automatically closes when exiting 'with' block
        """
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically disconnect when exiting 'with' block."""
        self.disconnect()


# =================================================================
# SECTION 3: CONVENIENCE FUNCTIONS
# =================================================================

def quick_analysis(entity_id: int, 
                   db_config: Optional[DatabaseConfig] = None,
                   output_dir: str = "output") -> Dict:
    """
    Quick single-entity analysis with automatic connection management.
    
    TUTORIAL: This is the easiest way to analyze a single entity.
    It handles all connection management automatically.
    
    Args:
        entity_id: Entity ID to analyze
        db_config: Database configuration (loads from env if None)
        output_dir: Output directory
        
    Returns:
        Analysis results dictionary
        
    Example:
        >>> # Make sure environment variables are set first!
        >>> results = quick_analysis(entity_id=1)
        >>> print(f"Status: {results['status']}")
    """
    
    if db_config is None:
        db_config = DatabaseConfig.from_env()
    
    with ESGMaterialityController(db_config, output_dir) as controller:
        results = controller.run_analysis(entity_id)
        controller.save_results(results)
        controller.export_to_csv(results)
    
    return results


def batch_analysis(entity_ids: List[int],
                   db_config: Optional[DatabaseConfig] = None,
                   output_dir: str = "output") -> Dict:
    """
    Quick batch analysis with automatic connection management.
    
    TUTORIAL: Easiest way to analyze multiple entities.
    
    Args:
        entity_ids: List of entity IDs to analyze
        db_config: Database configuration (loads from env if None)
        output_dir: Output directory
        
    Returns:
        Batch results dictionary
        
    Example:
        >>> results = batch_analysis([1, 4, 24, 126])
        >>> print(f"Success rate: {results['summary']['success_rate']:.1%}")
    """
    
    if db_config is None:
        db_config = DatabaseConfig.from_env()
    
    with ESGMaterialityController(db_config, output_dir) as controller:
        batch_results = controller.run_batch_analysis(entity_ids)
    
    return batch_results


# =================================================================
# SECTION 4: EXAMPLE USAGE
# =================================================================

if __name__ == "__main__":
    """
    TUTORIAL: Example usage demonstrating different approaches.
    
    Run this script to see the controller in action!
    
    Before running, make sure to:
    1. Set up PostgreSQL database with ESG data
    2. Set environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
    3. Or modify DatabaseConfig parameters below
    """
    
    print("="*70)
    print("ESG MATERIALITY ANALYSIS - EXAMPLE USAGE")
    print("="*70)
    
    # ==========================================
    # APPROACH 1: Manual Configuration (Development)
    # ==========================================
    print("\n Approach 1: Manual configuration")
    
    db_config = DatabaseConfig(
        host="localhost",
        database="esg_database",
        user="postgres",
        password="your_password",
        port=5432
    )
    
    controller = ESGMaterialityController(db_config, output_dir="output")
    
    try:
        # Connect
        controller.connect()
        
        # Analyze single entity
        results = controller.run_analysis(entity_id=1)
        
        # Save results
        controller.save_results(results)
        controller.export_to_csv(results)
        
        # Disconnect
        controller.disconnect()
        
    except Exception as e:
        print(f"Analysis failed: {e}")
    
    # ==========================================
    # APPROACH 2: Environment Variables (Production)
    # ==========================================
    print("\n\n Approach 2: Environment variables")
    
    # Load config from environment
    db_config = DatabaseConfig.from_env()
    
    # Use context manager (recommended)
    with ESGMaterialityController(db_config) as controller:
        results = controller.run_analysis(entity_id=1)
        controller.save_results(results)
        controller.export_to_csv(results)
    
    # ==========================================
    # APPROACH 3: Quick Analysis (Simplest)
    # ==========================================
    print("\n\n Approach 3: Quick analysis function")
    
    results = quick_analysis(entity_id=1)
    print(f"Analysis status: {results['status']}")
    
    # ==========================================
    # APPROACH 4: Batch Analysis
    # ==========================================
    print("\n\n Approach 4: Batch analysis")
    
    batch_results = batch_analysis(entity_ids=[1, 4, 24, 126])
    print(f"Batch success rate: {batch_results['summary']['success_rate']:.1%}")
    
    print("\n" + "="*70)
    print("✅EXAMPLES COMPLETE - Check output/ directory for results")
    print("="*70)