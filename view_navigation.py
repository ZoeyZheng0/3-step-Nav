# streamlit run view_navigation.py
import streamlit as st
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timezone
import cv2
from PIL import Image, ImageDraw
import os

# Configuration
st.set_page_config(
    page_title="Open-Nav Visualization Dashboard",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main folders
LOGS_DIR = Path("/home2/zoey/code/vlnce/Open-Nav/logs")
EVAL_RESULTS_DIR = LOGS_DIR / "eval_results"
IMAGE_SHOW_DIR = Path("/home2/zoey/code/vlnce/Open-Nav/image_show")

def load_experiment_data(experiment_name):
    """Load experiment debug data and metrics"""
    experiment_data = {}
    
    # Load debug.json
    debug_path = EVAL_RESULTS_DIR / experiment_name / "debug.json"
    if debug_path.exists():
        with open(debug_path, 'r') as f:
            experiment_data['debug'] = json.load(f)
    
    # Load stats json file
    stats_files = list((EVAL_RESULTS_DIR / experiment_name).glob("stats_ep_ckpt_*.json"))
    if stats_files:
        # Use the first stats file found
        with open(stats_files[0], 'r') as f:
            experiment_data['stats'] = json.load(f)
    
    return experiment_data


def display_navigation_steps(episode_info, episode_id):
    
    steps = episode_info.get('steps', [])
    if not steps or len(steps) == 0:
        st.warning("No step data available")
        return
    
    # Step selector - handle single step case
    if len(steps) == 1:
        step_index = 0
        st.info("Only one step available")
    else:
        step_display = st.slider("Select Step", 1, len(steps), 1)
        step_index = step_display - 1  # Convert from 1-based to 0-based indexing
    step_data = steps[step_index]
    
    st.markdown(f"**Step {step_data['step_index']}**: {step_data.get('current_action', 'N/A')}")
    
    # Display viewpoints
    viewpoints = step_data.get('viewpoints', {})
    if viewpoints:
        st.markdown("#### Viewpoint Images")
        
        # Organize viewpoints in rows
        cols_per_row = 4
        viewpoint_ids = sorted(viewpoints.keys())
        
        for i in range(0, len(viewpoint_ids), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(viewpoint_ids):
                    vp_id = viewpoint_ids[i + j]
                    vp_data = viewpoints[vp_id]
                    
                    with col:
                        # Determine label with checkmark for chosen viewpoint
                        if vp_data.get('is_chosen', False):
                            label = f"VP {vp_id} ✅"
                        else:
                            label = f"VP {vp_id}"
                        
                        # Display RGB image from base64 or file path
                        if 'rgb_base64' in vp_data and vp_data['rgb_base64']:
                            # Use base64 image if available
                            st.image(vp_data['rgb_base64'], caption=label, use_container_width=True)
                        else:
                            # Fallback to file path method
                            rgb_path = Path(EVAL_RESULTS_DIR) / "episode_images" / str(episode_id) / vp_data['rgb_path']
                            try:
                                st.image(str(rgb_path), caption=label, use_container_width=True)
                            except:
                                st.error(f"Image not found: {vp_data['rgb_path']}")
    

def display_episode_info(debug_data, stats_data, selected_episode_id):
    """Display information for a selected episode"""
    
    # Find the selected episode in debug data
    episode_info = None
    for ep in debug_data.get('episodes', []):
        if ep['episode_id'] == selected_episode_id:
            episode_info = ep
            break
    
    if episode_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Episode Information**")
            st.info(f"Episode ID: {episode_info['episode_id']}")
            # Extract just the scene name from the full path
            scene_name = episode_info['scene_id'].split('/')[-1].replace('.glb', '')
            st.info(f"Scene ID: {scene_name}")
        
        with col2:
            st.text_area("Instruction", episode_info['instruction'], height=100)
        
        # Display metrics if available
        if stats_data and selected_episode_id in stats_data:
            st.markdown("### 📊 Episode Metrics")
            metrics = stats_data[selected_episode_id]
            
            # Create metric columns
            metric_cols = st.columns(7)
            
            with metric_cols[0]:
                st.metric("Success", "✅" if metrics.get('success', 0) == 1 else "❌")
            
            with metric_cols[1]:
                st.metric("Oracle Success", "✅" if metrics.get('oracle_success', 0) == 1 else "❌")
            
            with metric_cols[2]:
                st.metric("SPL", f"{metrics.get('spl', 0):.3f}")
            
            with metric_cols[3]:
                st.metric("NDTW", f"{metrics.get('ndtw', 0):.3f}")
            
            with metric_cols[4]:
                st.metric("Steps", f"{metrics.get('steps_taken', 0):.0f}")
            
            with metric_cols[5]:
                st.metric("Distance to Goal", f"{metrics.get('distance_to_goal', 0):.2f}m")
            
            with metric_cols[6]:
                st.metric("Path Length", f"{metrics.get('path_length', 0):.2f}m")
            
            # Detailed metrics table
            with st.expander("View All Metrics"):
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df, use_container_width=True)
        
        # Display step-by-step navigation if available
        if 'steps' in episode_info and episode_info['steps']:
            display_navigation_steps(episode_info, selected_episode_id)

def display_experiment_summary(debug_data, stats_data):
    """Display overall experiment summary"""
    st.subheader("📈 Experiment Summary")
    
    if stats_data:
        # Calculate aggregate metrics
        total_episodes = len(stats_data)
        successful_episodes = sum(1 for m in stats_data.values() if m.get('success', 0) == 1)
        oracle_successful_episodes = sum(1 for m in stats_data.values() if m.get('oracle_success', 0) == 1)
        avg_spl = np.mean([m.get('spl', 0) for m in stats_data.values()])
        avg_ndtw = np.mean([m.get('ndtw', 0) for m in stats_data.values()])
        avg_steps = np.mean([m.get('steps_taken', 0) for m in stats_data.values()])
        avg_distance = np.mean([m.get('distance_to_goal', 0) for m in stats_data.values()])
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Episodes", total_episodes)
            st.metric("Success Rate", f"{(successful_episodes/total_episodes)*100:.1f}%")
            st.metric("Oracle Success Rate", f"{(oracle_successful_episodes/total_episodes)*100:.1f}%")
        
        with col2:
            st.metric("Avg SPL", f"{avg_spl:.3f}")
            st.metric("Avg NDTW", f"{avg_ndtw:.3f}")
        
        with col3:
            st.metric("Avg Steps", f"{avg_steps:.1f}")
            st.metric("Avg Distance to Goal", f"{avg_distance:.2f}m")
        
        with col4:
            # Success distribution pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Success', 'Failure'],
                values=[successful_episodes, total_episodes - successful_episodes],
                hole=0.3,
                marker_colors=['#00cc44', '#ff4444']
            )])
            fig.update_layout(
                title="Success Distribution",
                height=200,
                showlegend=True,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Episode metrics visualization
        st.markdown("### 📊 Episode Metrics Distribution")
        
        # Create dataframe for visualization
        episodes_list = []
        for ep_id, metrics in stats_data.items():
            episodes_list.append({
                'Episode ID': ep_id,
                'Success': metrics.get('success', 0),
                'Oracle Success': metrics.get('oracle_success', 0),
                'SPL': metrics.get('spl', 0),
                'NDTW': metrics.get('ndtw', 0),
                'Steps': metrics.get('steps_taken', 0),
                'Distance to Goal': metrics.get('distance_to_goal', 0),
                'Path Length': metrics.get('path_length', 0)
            })
        
        episodes_df = pd.DataFrame(episodes_list)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # SPL distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=episodes_df['SPL'],
                nbinsx=20,
                name='SPL Distribution',
                marker_color='blue'
            ))
            fig.update_layout(
                title="SPL Distribution",
                xaxis_title="SPL",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distance to goal distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=episodes_df['Distance to Goal'],
                nbinsx=20,
                name='Distance Distribution',
                marker_color='green'
            ))
            fig.update_layout(
                title="Distance to Goal Distribution",
                xaxis_title="Distance (m)",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Full episode table
        with st.expander("View All Episodes"):
            st.dataframe(
                episodes_df.style.format({
                    'Success': lambda x: '✅' if x == 1 else '❌',
                    'Oracle Success': lambda x: '✅' if x == 1 else '❌',
                    'SPL': '{:.3f}',
                    'NDTW': '{:.3f}',
                    'Steps': '{:.0f}',
                    'Distance to Goal': '{:.2f}',
                    'Path Length': '{:.2f}'
                }),
                use_container_width=True
            )

def main():
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        # Select experiment
        st.subheader("Experiment Selection")
        # Get experiments sorted by modification time (latest first)
        experiment_dirs = [d for d in EVAL_RESULTS_DIR.iterdir() if d.is_dir()]
        if experiment_dirs:
            # Sort by modification time, latest first
            experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            experiments = [d.name for d in experiment_dirs]
            
            # Create display names with (Latest) indicator
            display_names = [f"{exp} (Latest)" if i == 0 else exp for i, exp in enumerate(experiments)]
            
            # Default to the latest experiment
            selected_display = st.selectbox("Select Experiment", display_names, index=0)
            
            # Extract actual experiment name (remove " (Latest)" if present)
            experiment = selected_display.replace(" (Latest)", "")
            
            # Load experiment data
            experiment_data = load_experiment_data(experiment)
            
            if experiment_data.get('debug'):
                st.success(f"Loaded experiment: {experiment}")
                
                # Show experiment timestamp
                exp_path = EVAL_RESULTS_DIR / experiment
                if exp_path.exists():
                    mod_time = datetime.fromtimestamp(exp_path.stat().st_mtime)
                    st.caption(f"Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Episode selection
                episodes = experiment_data['debug'].get('episodes', [])
                if episodes:
                    st.subheader("Episode Selection")
                    episode_ids = [ep['episode_id'] for ep in episodes]
                    selected_episode = st.selectbox("Select Episode", episode_ids)
                    
                    # Display experiment info
                    st.markdown("---")
                    st.markdown("**Experiment Info**")
                    st.info(f"Total Episodes: {len(episodes)}")
                    
                    if experiment_data.get('stats'):
                        success_count = sum(1 for m in experiment_data['stats'].values() 
                                          if m.get('success', 0) == 1)
                        oracle_success_count = sum(1 for m in experiment_data['stats'].values() 
                                          if m.get('oracle_success', 0) == 1)
                        st.info(f"Success Rate: {(success_count/len(episodes))*100:.1f}%")
                        st.info(f"Oracle Success Rate: {(oracle_success_count/len(episodes))*100:.1f}%")
                else:
                    st.warning("No episodes found in debug.json")
            else:
                st.warning(f"No debug.json found for {experiment}")
        else:
            st.warning("No experiments found in eval_results directory")
    
    # Main content area
    if 'experiment_data' in locals() and experiment_data:
        tab1, tab2 = st.tabs(["Episode View", "Experiment Summary"])
        
        with tab1:
            if 'selected_episode' in locals():
                display_episode_info(
                    experiment_data.get('debug', {}),
                    experiment_data.get('stats', {}),
                    selected_episode
                )
        
        with tab2:
            display_experiment_summary(
                experiment_data.get('debug', {}),
                experiment_data.get('stats', {})
            )
    else:
        st.info("Select an experiment from the sidebar to begin visualization")

if __name__ == "__main__":
    main()