# Deployments

## Release schedule
Services deploy continuously on weekdays between 10:00 and 16:00. There are no production deployments on Fridays after 14:00 or during the last week of December.

## Rollout
New versions go to 5% of traffic first, then 25%, then everyone, with 15 minutes between steps. Automatic rollback triggers if the error rate rises above 2%.

## Database migrations
Migrations must be backwards compatible with the previous application version. Destructive changes such as dropping a column ship in a separate release at least one week later.
