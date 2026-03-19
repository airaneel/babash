from .client.mcp_server import main as mcp_server

# Export mcp_server as the default entry point for babash
listen = mcp_server
